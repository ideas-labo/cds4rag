import re

def clean_json_string(s):
    s = s.strip().lstrip('\ufeff')
    s = s.replace('\\n', '\n').replace('\\t', '\t')
    s = re.sub(r'\\(?!["\\/bfnrt])', '', s)
    return s

def verify_json_format(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return True
    except Exception as e:
        return False

def save_vectorstore(vectorstore, filepath="vectorstore.pickle"):
    import pickle
    try:
        with open(filepath, "wb") as f:
            pickle.dump(vectorstore, f)
        return True
    except Exception as e:
        if hasattr(vectorstore, 'save_local'):
            try:
                directory = filepath.replace('.pickle', '_directory')
                vectorstore.save_local(directory)
                return True
            except Exception as e2:
                pass
        return False

def load_vectorstore(filepath="vectorstore.pickle"):
    import pickle
    import os
    
    # 检查文件是否存在
    if os.path.exists(filepath):
        try:
            with open(filepath, "rb") as f:
                vectorstore = pickle.load(f)
            return vectorstore
        except Exception as e:
            pass

    directory = filepath.replace('.pickle', '_directory')
    if os.path.exists(directory):
        try:
            if os.path.exists(os.path.join(directory, 'index.faiss')):
                from langchain_community.vectorstores import FAISS
                vectorstore = FAISS.load_local(directory, embedding=None)  
                return vectorstore
            elif os.path.exists(os.path.join(directory, 'chroma')):
                from langchain_community.vectorstores import Chroma
                vectorstore = Chroma(persist_directory=directory, embedding_function=None)
                return vectorstore
        except Exception as e:
            pass
    
    return None