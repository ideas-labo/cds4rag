import json
import time
from langchain_ollama import ChatOllama
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_ollama import OllamaEmbeddings
from langchain.docstore.document import Document
from word_chunker import chunk_documents_by_words
from utils import TokenCounter, calculate_retrieval_metrics,  calculate_generation_metrics
from utils import create_vectorstore
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

token_counter = TokenCounter()

class TrackedOllamaEmbeddings(OllamaEmbeddings):
    
    def embed_query(self, text: str) -> list[float]:
        token_counter.add_embedding_tokens(text)
        return super().embed_query(text)
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        for text in texts:
            token_counter.add_embedding_tokens(text)
        
        return super().embed_documents(texts)

class TrackedChatOllama(ChatOllama):
    
    def stream(self, messages):
        if isinstance(messages, list):
            input_text = "\n".join([
                f"{msg.get('role', 'user')}: {msg.get('content', '')}" 
                for msg in messages if isinstance(msg, dict)
            ])
        else:
            input_text = str(messages)
        
        output_chunks = []
        for chunk in super().stream(messages):
            output_chunks.append(chunk)
            yield chunk
        
        output_text = "".join([chunk.content for chunk in output_chunks])

        token_counter.add_llm_tokens(input_text, output_text)

class Config:
    DATABASE_TYPE = "duckdb"
    OLLAMA_BASE_URL = "http://localhost:11434"
    EMBEDDING_MODEL = "bge-m3:567m"
    LLM_MODEL = "llama3.1:8b"
    
    VECTOR_INDEX_NAME = "vector"
    NODE_LABEL = "Chunk"
    EMBEDDING_PROPERTY = "embedding"
    TEXT_PROPERTY = "text"
    DATABASE_NAME = "duckdb"
    
    CHUNK_SIZE = 256
    CHUNK_OVERLAP = 32

    RETRIEVER_K = 10

    Question_ratio = 0.3
    Question_difficulty = 0
    Corpus_scale = 0
    Dataset_category = 0
    
    embedding_temperature = 0.8
    embedding_num_ctx = 2048
    embedding_top_k = 0
    embedding_repeat_penalty = 1.1
    
    chat_temperature = 0.4
    chat_num_ctx = 4096
    chat_top_k = 40
    chat_repeat_penalty = 1.2

    CONTEXT_INPUT_PATH = "./datasets/unique_contexts/agriculture_unique_contexts.json"
    CONTEXT_OUTPUT_PATH = "./datasets/unique_contexts/agriculture_chunked.json"
    QA_FILE_PATH = "./datasets/agriculture.jsonl"
    CONTEXT_INPUT_PATH_other = None
    RELEVANCE_THRESHOLD = 0.5


def get_other_corpus(main_category, corpus_scale, categories):
    """
    Merge corpora
    main_category: Main category
    corpus_scale: Number of other corpora to mix in
    categories: All available categories
    """
    other_categories = [cat for cat_id, cat in categories.items() if cat_id != main_category]
    selected_categories = np.random.choice(other_categories, corpus_scale, replace=False)
    
    output_paths = []
    for category in selected_categories:
        other_path = f"./datasets/unique_contexts/{category}_unique_contexts.json"
        output_paths.append(other_path)
    
    return output_paths

def load_jsonl(file_path):
    documents = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            documents.append(json.loads(line.strip()))
    return documents

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def evaluate_retrieval(retriever, questions, embedding_model):
    """
    Function to evaluate retrieval quality only, not involving generation

    Args:
        retriever: Retriever instance
        questions: List of questions
        embedding_model: Embedding model

    Returns:
        retrieval_results: List containing detailed retrieval information
        avg_retrieval_metrics: Average retrieval metrics
        token_usage_summary: Token usage summary
        question_token_usage: Token usage details for each question
    """
    token_counter.reset()
    
    retrieval_results = []
    
    retrieval_metrics = {
        'total_questions': 0,
        'mrr_sum': 0.0,
        'ndcg_sum': 0.0,
        'context_similarity_sum': 0.0,
        'relevant_docs_sum': 0.0
    }
    
    question_token_usage = []
    
    for example in questions:
        question = example["input"]
        true_context = example.get("context", "")
        
        start_embedding_tokens = token_counter.embedding_tokens
        
        retrieved_docs = retriever.invoke(question)
        retrieved_context = format_docs(retrieved_docs)
        
        detailed_metrics = calculate_retrieval_metrics(retrieved_docs, true_context, embedding_model, question)
        
        retrieval_metrics['mrr_sum'] += detailed_metrics['mrr']
        retrieval_metrics['ndcg_sum'] += detailed_metrics['ndcg']
        retrieval_metrics['context_similarity_sum'] += detailed_metrics['context_similarity']
        retrieval_metrics['relevant_docs_sum'] += detailed_metrics['relevant_docs_count']
        retrieval_metrics['total_questions'] += 1
        
        end_embedding_tokens = token_counter.embedding_tokens
        question_embedding_tokens = end_embedding_tokens - start_embedding_tokens
        
        question_token_usage.append({
            'question_id': len(question_token_usage),
            'question': question[:100] + "..." if len(question) > 100 else question,
            'embedding_tokens': question_embedding_tokens,
            'total_tokens': question_embedding_tokens
        })
        
        retrieval_results.append({
            'question': question,
            'retrieved_docs': [{'content': doc.page_content, 'metadata': doc.metadata} for doc in retrieved_docs],
            'retrieved_context': retrieved_context,
            'true_context': true_context,
            'mrr': detailed_metrics['mrr'],
            'ndcg': detailed_metrics['ndcg'],
            'context_similarity': detailed_metrics['context_similarity'],
            'token_usage': {
                'embedding_tokens': question_embedding_tokens,
                'total_tokens': question_embedding_tokens
            }
        })
    
    total = retrieval_metrics['total_questions']
    if total > 0:
        avg_retrieval_metrics = {
            'mean_reciprocal_rank': retrieval_metrics['mrr_sum'] / total,
            'ndcg': retrieval_metrics['ndcg_sum'] / total,
            'context_similarity': retrieval_metrics['context_similarity_sum'] / total,
            'avg_relevant_docs': retrieval_metrics['relevant_docs_sum'] / total
        }
    else:
        avg_retrieval_metrics = {
            'mean_reciprocal_rank': 0,
            'ndcg': 0,
            'context_similarity': 0,
            'avg_relevant_docs': 0
        }
    
    token_usage_summary = token_counter.get_summary()
    
    return retrieval_results, avg_retrieval_metrics, token_usage_summary, question_token_usage


def evaluate_generation(retrieval_results, llm, questions, embedding_model, category):
    """
    Evaluate generation quality based on existing retrieval results

    Args:
        retrieval_results: List of retrieval results
        llm: Large language model instance
        questions: List of questions
        embedding_model: Embedding model

    Returns:
        predictions: List of generated answers
        references: List of reference answers
        generation_results: List containing detailed generation information
        generation_metrics: Generation quality metrics
        token_usage_summary: Token usage summary
        question_token_usage: Token usage details for each question
    """
    token_counter.reset()
    
    predictions = []
    references = []
    generation_results = []
    
    question_token_usage = []
    
    for i, example in enumerate(questions):
        if i >= len(retrieval_results):
            break
            
        retrieval_result = retrieval_results[i]
        question = example["input"]
        true_answer = example["answers"]
        retrieved_context = retrieval_result["retrieved_context"]
        
        start_llm_tokens = token_counter.llm_input_tokens + token_counter.llm_output_tokens

        if category == "hotpot":
            formatted_prompt = [
                {"role": "system", "content": """You are a helpful, respectful and honest assistant.
            Always answer as helpfully as possible, while being safe.
            Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.
            Please ensure that your responses are socially unbiased and positive in nature.
            If you need include the answer just output the answer, no need to explain."""},
                {"role": "user", "content": f"""[document]: {retrieved_context}
            [conversation]: {question}. Answer with no more than 5 words. 
            Answer solely based on the provided context. If information is absent, state 'I do not know'."""}
            ]
        else:
            formatted_prompt =[
                {"role": "system", "content": """You are a helpful, respectful and honest assistant.
                Always answer as helpfully as possible, while being safe.
                Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.
                Please ensure that your responses are socially unbiased and positive in nature.
                If you need include the answer just output the answer, no need to explain.
                If you don't know the answer to a question, please don't share false information."""},
                {"role": "user", "content": f"""[document]: {retrieved_context}
            [conversation]: {question}. Answer with no more than 150 words. 
            Answer solely based on the provided context. If information is absent, state 'I do not know'."""}
            ]

        predicted_answer = ""
        for chunk in llm.stream(formatted_prompt):
            predicted_answer += chunk.content
        
        predictions.append(predicted_answer)
        references.append(true_answer)
        
        
        end_llm_tokens = token_counter.llm_input_tokens + token_counter.llm_output_tokens
        question_llm_tokens = end_llm_tokens - start_llm_tokens
        
        question_token_usage.append({
            'question_id': len(question_token_usage),
            'question': question[:100] + "..." if len(question) > 100 else question,
            'llm_tokens': question_llm_tokens,
            'total_tokens': question_llm_tokens
        })
    
    generation_metrics = calculate_generation_metrics(predictions, references, llm)
    
    token_usage_summary = token_counter.get_summary()
    
    return predictions, references, generation_results, generation_metrics, token_usage_summary, question_token_usage

def run_rag_evaluation(config, only_retrieval=False, saved_vectorstore=None, formatted_prompt=None):
    categories = {
        0: "agriculture",
        1: "biography", 
        2: "hotpot",
        3: "bioasq"
    }
    category = categories[config.Dataset_category]
    config.CONTEXT_INPUT_PATH = f"./datasets/unique_contexts/{category}_unique_contexts.json"
    config.CONTEXT_OUTPUT_PATH = f"./datasets/unique_contexts/{category}_chunked.json"
    config.QA_FILE_PATH = f"./datasets/{category}.jsonl"
    
    if saved_vectorstore is not None:
        vectorstore = saved_vectorstore
        retriever = vectorstore.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": config.RETRIEVER_K}
        )
        chunk_time = 0
        build_time = 0
        embedding_model = TrackedOllamaEmbeddings(
            base_url=config.OLLAMA_BASE_URL, 
            model=config.EMBEDDING_MODEL,
            temperature=config.embedding_temperature,
            top_k=config.embedding_top_k,
            num_ctx=config.embedding_num_ctx,
            repeat_penalty=config.embedding_repeat_penalty
        )
    else:
        chunk_start_time = time.time()
        documents_all = []
        if config.CONTEXT_INPUT_PATH_other:
            for path in config.CONTEXT_INPUT_PATH_other:
                documents = chunk_documents_by_words(
                input_path=path,
                output_path=config.CONTEXT_OUTPUT_PATH,
                chunk_size=config.CHUNK_SIZE,
                chunk_overlap=config.CHUNK_OVERLAP)
                documents_all.extend(documents)
        
        documents = chunk_documents_by_words(
            input_path=config.CONTEXT_INPUT_PATH,
            output_path=config.CONTEXT_OUTPUT_PATH,
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP)
        documents_all.extend(documents)

        chunk_end_time = time.time()
        chunk_time = chunk_end_time - chunk_start_time
        build_start_time = time.time()

        docs = [Document(page_content=doc["contents"], metadata={"id": doc["id"]}) for doc in documents_all]
        
        embedding_model = TrackedOllamaEmbeddings(
            base_url=config.OLLAMA_BASE_URL, 
            model=config.EMBEDDING_MODEL,
            temperature=config.embedding_temperature,
            top_k=config.embedding_top_k,
            num_ctx=config.embedding_num_ctx,
            repeat_penalty=config.embedding_repeat_penalty
        )
        
        vectorstore = create_vectorstore(config, docs, embedding_model)

        retriever = vectorstore.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": config.RETRIEVER_K}
        )
        
        build_end_time = time.time()
        build_time = build_end_time - build_start_time


    llm = TrackedChatOllama(
        base_url=config.OLLAMA_BASE_URL, 
        model=config.LLM_MODEL,
        temperature=config.chat_temperature,
        top_k=config.chat_top_k,
        num_ctx=config.chat_num_ctx,
        repeat_penalty=config.chat_repeat_penalty,
        seed=1
    )

    test_start_time = time.time()
    question_ans = load_jsonl(config.QA_FILE_PATH)
    selected_questions = question_ans[:100]

    print(f"Question count: {len(selected_questions)}")
 
    
    retrieval_results, retrieval_metrics, retrieval_token_usage, retrieval_question_tokens = evaluate_retrieval(
        retriever, 
        selected_questions,
        embedding_model
    )
    
    results = {
        "retrieval_metrics": retrieval_metrics,
        "retrieval_token_usage": retrieval_token_usage,
        "retrieval_question_tokens": retrieval_question_tokens
    }
    
    if only_retrieval:
        test_end_time = time.time()
        test_time = test_end_time - test_start_time
        
        results.update({
            "chunk_time_seconds": chunk_time,
            "build_time_seconds": build_time,
            "test_time_seconds": test_time,
            "total_time_seconds": chunk_time + build_time + test_time,
        })
        
        return vectorstore, results
    
    predictions, references, _, generation_metrics, generation_token_usage, generation_question_tokens = evaluate_generation(
        retrieval_results,
        llm,
        selected_questions,
        embedding_model,
        category
    )
    
    avg_similarity = 0
    test_end_time = time.time()
    test_time = test_end_time - test_start_time
    
    results.update({
        "average_similarity": avg_similarity,
        "chunk_time_seconds": chunk_time,
        "build_time_seconds": build_time,
        "test_time_seconds": test_time,
        "total_time_seconds": chunk_time + build_time + test_time,
        "generation_metrics": generation_metrics,
        "generation_token_usage": generation_token_usage,
        "generation_question_tokens": generation_question_tokens,
        "token_statistics": {
            "avg_tokens_per_question": (retrieval_token_usage['total_tokens'] + generation_token_usage['total_tokens']) / len(selected_questions) if selected_questions else 0,
            "avg_embedding_tokens_per_question": retrieval_token_usage['embedding_tokens'] / len(selected_questions) if selected_questions else 0,
            "avg_llm_tokens_per_question": generation_token_usage['total_llm_tokens'] / len(selected_questions) if selected_questions else 0,
            "embedding_to_llm_ratio": retrieval_token_usage['embedding_tokens'] / generation_token_usage['total_llm_tokens'] if generation_token_usage['total_llm_tokens'] > 0 else 0
        }
    })
    
    return results


def calculate_similarity(predictions, references, embedding_model):
    pred_embeddings = [embedding_model.embed_query(pred) for pred in predictions]
    ref_embeddings = [embedding_model.embed_query(ref[0]) for ref in references]
    
    similarities = []
    for pred_emb, ref_emb in zip[tuple](pred_embeddings, ref_embeddings):
        pred_emb = np.array(pred_emb).reshape(1, -1)
        ref_emb = np.array(ref_emb).reshape(1, -1)
        similarity = cosine_similarity(pred_emb, ref_emb)[0][0]
        similarities.append(similarity)
    
    return np.mean(similarities)

def save_results_to_json(results, output_file="results.json"):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

def create_chroma_config():
    config = Config()
    config.DATABASE_TYPE = "chroma"
    return config

def create_faiss_config():
    config = Config()
    config.DATABASE_TYPE = "faiss"
    return config

def create_duckdb_config():
    config = Config()
    config.DATABASE_TYPE = "duckdb"
    return config


if __name__ == "__main__":
    
    config = Config()

    configs = {
        "chroma": create_chroma_config(),
        "faiss": create_faiss_config(),
        "duckdb": create_duckdb_config()
    }
    
    database_choice = "duckdb"
    config = configs[database_choice]
    vectorstore, retrieval_only_results = run_rag_evaluation(config, only_retrieval=True)
    full_results = run_rag_evaluation(config, only_retrieval=False, saved_vectorstore=vectorstore)
    print(full_results['generation_metrics'])
