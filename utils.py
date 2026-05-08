import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.vectorstores import Chroma, FAISS, DuckDB
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import tiktoken 

def calculate_llmaaj_correctness(prediction, reference, llm):
    if not prediction or not reference:
        return 0.0

    if isinstance(reference, list):
        reference = reference[0] if reference else ""

    prompt = f"""As a professional evaluator, assess the correctness of the generated answer compared to the reference answer.
    Reference answer: {reference}
    Generated answer: {prediction}

    Based on whether the generated answer contains the key information from the reference answer, whether it has any incorrect information, and its overall accuracy, provide a score between 0 and 1.
    1 means completely correct, 0 means completely wrong. Return only the numeric score without any explanation."""

    try:
        response = ""
        for chunk in llm.stream([{"role": "user", "content": prompt}]):
            response += chunk.content

        import re
        score_match = re.search(r'(\d+\.\d+|\d+)', response)
        if score_match:
            score = float(score_match.group(1))
            return min(max(score, 0.0), 1.0)
        else:
            return calculate_answer_f1_score(prediction, reference)
    except Exception as e:
        print(f"LLMaaJ评估失败: {e}")
        return calculate_answer_f1_score(prediction, reference)

class TokenCounter:
    
    def __init__(self):
        self.reset()
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base") 
        except:
            self.tokenizer = None
    
    def reset(self):
        self.embedding_tokens = 0
        self.llm_input_tokens = 0
        self.llm_output_tokens = 0
        self.total_embedding_calls = 0
        self.total_llm_calls = 0
    
    def estimate_tokens(self, text):
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            return max(len(text) // 3, len(text.split()))
    
    def add_embedding_tokens(self, text):
        tokens = self.estimate_tokens(text)
        self.embedding_tokens += tokens
        self.total_embedding_calls += 1
        return tokens
    
    def add_llm_tokens(self, input_text, output_text):
        input_tokens = self.estimate_tokens(input_text)
        output_tokens = self.estimate_tokens(output_text)
        
        self.llm_input_tokens += input_tokens
        self.llm_output_tokens += output_tokens
        self.total_llm_calls += 1
        
        return input_tokens, output_tokens
    
    def get_summary(self):
        return {
            'embedding_tokens': self.embedding_tokens,
            'llm_input_tokens': self.llm_input_tokens,
            'llm_output_tokens': self.llm_output_tokens,
            'total_llm_tokens': self.llm_input_tokens + self.llm_output_tokens,
            'total_tokens': self.embedding_tokens + self.llm_input_tokens + self.llm_output_tokens,
            'embedding_calls': self.total_embedding_calls,
            'llm_calls': self.total_llm_calls
        }


def calculate_retrieval_metrics(retrieved_docs, true_context, embedding_model, question=None):
    """
    计算3个核心检索评估指标：MRR, NDCG, Context Similarity，基于文本匹配而非嵌入向量
    """
    metrics = {
        'mrr': 0.0,
        'ndcg': 0.0,
        'context_similarity': 0.0,
        'best_match_position': -1,
        'relevant_docs_count': 0
    }
    
    if not true_context or not retrieved_docs:
        return metrics
    
    true_context = true_context.strip().lower()
    
    relevance_scores = []
    similarities = []

    for i, doc in enumerate(retrieved_docs):
        doc_content = doc.page_content.strip().lower()

        is_substring = doc_content in true_context or true_context in doc_content
        
        words_doc = set(doc_content.split())
        words_true = set(true_context.split())
        overlap_ratio = len(words_doc.intersection(words_true)) / max(len(words_doc), 1) if words_doc else 0
        
        is_relevant = overlap_ratio >= 0.8 or is_substring
        relevance_scores.append(1 if is_relevant else 0)
        similarities.append(overlap_ratio)
    
    for i, is_relevant in enumerate(relevance_scores):
        if is_relevant:
            metrics['mrr'] = 1.0 / (i + 1)
            metrics['best_match_position'] = i + 1
            break

    def calculate_dcg(relevance_scores):
        dcg = 0.0
        for i, rel in enumerate(relevance_scores):
            if rel > 0:
                dcg += rel / np.log2(i + 2)
        return dcg
    
    dcg = calculate_dcg(relevance_scores)
    ideal_relevance = sorted(relevance_scores, reverse=True)
    idcg = calculate_dcg(ideal_relevance)
    
    if idcg > 0:
        metrics['ndcg'] = dcg / idcg
    
    if similarities:
        metrics['context_similarity'] = sum(similarities) / len(similarities)
    
    # 相关文档数量
    metrics['relevant_docs_count'] = sum(relevance_scores)
    
    return metrics

def calculate_retrieval_metrics_embedding(retrieved_docs, true_context, embedding_model, question):
    metrics = {
        'mrr': 0.0,
        'ndcg': 0.0,
        'context_similarity': 0.0,
        'best_match_position': -1,
        'relevant_docs_count': 0
    }
    
    if not true_context or not retrieved_docs:
        return metrics
    
    true_embedding = embedding_model.embed_query(true_context)
    true_embedding = np.array(true_embedding).reshape(1, -1)
    
    relevance_scores = []
    
    for i, doc in enumerate(retrieved_docs):
        doc_embedding = embedding_model.embed_query(doc.page_content)
        doc_embedding = np.array(doc_embedding).reshape(1, -1)
        similarity = cosine_similarity(true_embedding, doc_embedding)[0][0]
        
        is_relevant = similarity >= 0.6
        relevance_scores.append(1 if is_relevant else 0)
    
    for i, is_relevant in enumerate(relevance_scores):
        if is_relevant:
            metrics['mrr'] = 1.0 / (i + 1)
            metrics['best_match_position'] = i + 1
            break

    def calculate_dcg(relevance_scores):
        dcg = 0.0
        for i, rel in enumerate(relevance_scores):
            if rel > 0:
                dcg += rel / np.log2(i + 2)
        return dcg
    
    dcg = calculate_dcg(relevance_scores)
    ideal_relevance = sorted(relevance_scores, reverse=True)
    idcg = calculate_dcg(ideal_relevance)
    
    if idcg > 0:
        metrics['ndcg'] = dcg / idcg
    
    retrieved_context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    retrieved_embedding = embedding_model.embed_query(retrieved_context)
    retrieved_embedding = np.array(retrieved_embedding).reshape(1, -1)
    metrics['context_similarity'] = cosine_similarity(true_embedding, retrieved_embedding)[0][0]
    
    metrics['relevant_docs_count'] = sum(relevance_scores)
    
    return metrics

def _tokenize_words(text):
    if not isinstance(text, str):
        text = str(text)
    import re
    text = text.lower().strip()
    tokens = re.findall(r'[\u4e00-\u9fff]|[a-z0-9]+', text)
    return tokens

def calculate_lexical_answer_correctness(prediction, reference):
    if not prediction or not reference:
        return 0.0
    
    if isinstance(reference, list):
        reference = reference[0] if reference else ""
    
    from collections import Counter
    
    pred_tokens = _tokenize_words(prediction)
    ref_tokens = _tokenize_words(reference)
    
    if not ref_tokens:
        return 0.0
    
    pred_counter = Counter(pred_tokens)
    ref_counter = Counter(ref_tokens)
    
    matched_tokens = 0
    for token, ref_count in ref_counter.items():
        matched_tokens += min(pred_counter.get(token, 0), ref_count)
    
    lexical_ac = matched_tokens / len(ref_tokens)
    return lexical_ac

def calculate_answer_precision(prediction, reference):
    if not prediction or not reference:
        return 0.0
    
    if isinstance(reference, list):
        reference = reference[0] if reference else ""
    
    from collections import Counter
    
    pred_tokens = _tokenize_words(prediction)
    ref_tokens = _tokenize_words(reference)
    
    if not pred_tokens:
        return 0.0
    
    pred_counter = Counter(pred_tokens)
    ref_counter = Counter(ref_tokens)
    
    matched_tokens = 0
    for token, pred_count in pred_counter.items():
        matched_tokens += min(pred_count, ref_counter.get(token, 0))
    
    precision = matched_tokens / len(pred_tokens)
    return precision

def calculate_answer_f1_score(prediction, reference):
    recall = calculate_lexical_answer_correctness(prediction, reference)
    precision = calculate_answer_precision(prediction, reference)
    
    if recall + precision == 0:
        return 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def calculate_generation_metrics(predictions, references, llm):
    if not predictions or not references:
        return {
            'lexical_answer_correctness': 0.0,
            'answer_precision': 0.0,
            'answer_f1_score': 0.0,
            'llmaaj_correctness': 0.0
        }
    
    lexical_ac_scores = []
    precision_scores = []
    f1_scores = []
    llmaaj_scores = []

    
    for i, (pred, ref) in enumerate(zip(predictions, references)):
        lexical_ac = calculate_lexical_answer_correctness(pred, ref)
        lexical_ac_scores.append(lexical_ac)
        
        precision = calculate_answer_precision(pred, ref)
        precision_scores.append(precision)
        
        f1 = calculate_answer_f1_score(pred, ref)
        f1_scores.append(f1)
        print("pred:", pred)
        print("ref:", ref)
        llmaaj = 0
        llmaaj_scores.append(llmaaj)
    
    return {
        'lexical_answer_correctness': np.mean(lexical_ac_scores) if lexical_ac_scores else 0.0,
        'answer_precision': np.mean(precision_scores) if precision_scores else 0.0,
        'answer_f1_score': np.mean(f1_scores) if f1_scores else 0.0,
        'llmaaj_correctness': np.mean(llmaaj_scores) if llmaaj_scores else 0.0,
        'individual_scores': {
            'lexical_ac': lexical_ac_scores,
            'precision': precision_scores,
            'f1': f1_scores,
            'llmaaj': llmaaj_scores
        }
    }


def create_vectorstore(config, docs, embedding_model):
    if config.DATABASE_TYPE == "chroma":        
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embedding_model,
        )
    
    elif config.DATABASE_TYPE == "faiss":
        vectorstore = FAISS.from_documents(
            documents=docs,
            embedding=embedding_model
        )
    
    elif config.DATABASE_TYPE == "duckdb":
        vectorstore = DuckDB.from_documents(
            documents=docs,
            embedding=embedding_model,
        )
    
    else:
        raise ValueError(f"Error: {config.DATABASE_TYPE}")
    
    return vectorstore