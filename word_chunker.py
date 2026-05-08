import json
from typing import List, Dict, Any
import chonkie
from tqdm import tqdm
import re

def chunk_documents_by_words(
    input_path: str,
    output_path: str,
    chunk_size: int = 100,
    chunk_overlap: int = 20,
    content_field: str = "contents" 
) -> List[Dict[str, Any]]:

    documents = []
    original_char_count = 0
    
    with open(input_path, "r", encoding="utf-8") as f:
        try:
            contexts = json.load(f)
            if isinstance(contexts, list):
                for i, text in enumerate(contexts):
                    if isinstance(text, str):
                        documents.append({content_field: text, "original_index": i})
                        original_char_count += len(text)
                    else:
                        print(f"Warn (index{i})")
            else:
                print("Warning: JSON file is not in list format")
        except json.JSONDecodeError:
            print("Warning: JSON decoding failed, incorrect file format")

    print(f"Loaded {len(documents)} documents, total {original_char_count} characters")
    
    chunker = chonkie.TokenChunker(
        tokenizer="word",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    print("Chunking...")
    chunked_documents = []
    current_chunk_id = 0
    chunked_char_count = 0
    
    for doc in tqdm(documents):
        content = doc[content_field]
        
        chunks = chunker.chunk(content)
        
        for chunk in chunks:
            chunked_doc = doc.copy()
            chunked_doc["id"] = f"{current_chunk_id}"
            chunked_doc[content_field] = chunk.text
            chunked_documents.append(chunked_doc)
            current_chunk_id += 1
            chunked_char_count += len(chunk.text)

    print("Saving chunked results...")
    with open(output_path, "w", encoding="utf-8") as f:
        for doc in chunked_documents:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print(f"Done! Processed {len(documents)} documents, generated {len(chunked_documents)} chunks.")
    print(f"Original char count: {original_char_count}, Chunked char count: {chunked_char_count}")
    if original_char_count > 0:
        print(f"Character retention rate: {chunked_char_count/original_char_count:.2%}")
    return chunked_documents

def _word_tokens(text: str) -> List[str]:
    return re.findall(r'\w+|[\u4e00-\u9fff]', text)

def _count_words(text: str) -> int:
    return len(_word_tokens(text))

def recursive_split_text(
    text: str,
    chunk_size: int = 100,
    chunk_overlap: int = 20,
    separators: List[str] = ("\n\n", "\n", ". ", " ", "")
) -> List[str]:
    """
    Recursively split using natural boundaries (paragraph/line/sentence/space), 
    continue subdivision if length is not met; 
    when further subdivision is not possible, apply token sliding window with overlap.
    """
    def _split(t: str, seps: List[str]) -> List[str]:
        if _count_words(t) <= chunk_size * 1.2:
            return [t]
        if not seps:
            tokens = _word_tokens(t)
            windows = []
            start = 0
            step = max(chunk_size - chunk_overlap, 1)
            while start < len(tokens):
                end = min(start + chunk_size, len(tokens))
                windows.append(" ".join(tokens[start:end]))
                if end == len(tokens):
                    break
                start += step
            return windows
        sep = seps[0]
        parts = t.split(sep) if sep else [t]
        out: List[str] = []
        for p in parts:
            p = p.strip()
            if not p:
                continue
            if _count_words(p) <= chunk_size * 1.2:
                out.append(p)
            else:
                out.extend(_split(p, seps[1:]))
        return out

    raw_chunks = _split(text, list(separators))

    merged: List[str] = []
    buf: List[str] = []
    count = 0
    for piece in raw_chunks:
        pc = _count_words(piece)
        if count + pc <= chunk_size or not buf:
            buf.append(piece)
            count += pc
        else:
            merged.append(" ".join(buf))
            tail_tokens = _word_tokens(merged[-1])
            overlap_tail = " ".join(tail_tokens[max(0, len(tail_tokens) - chunk_overlap):])
            buf = [overlap_tail, piece] if overlap_tail else [piece]
            count = _count_words(" ".join(buf))
    if buf:
        merged.append(" ".join(buf))
    return merged

if __name__ == "__main__":
    chunks = chunk_documents_by_words(
        input_path="./datasets/unique_contexts/cs_unique_contexts.json",
        output_path="./datasets/unique_contexts/cs_word_chunked.jsonl",
        chunk_size=100,
        chunk_overlap=20,
        content_field="contents"
    )
