
# CDS4RAG: Cyclic Dual-Sequential Hyperparameter Optimization for RAG

This repository contains the data and code for the following paper: 
> Pengzhou Chen and Tao Chen. Cds4rag: Cyclic dual-sequential hyperparameter optimization
for rag. In Proceedings of the 35th International Joint Conference on Artificial Intelligence,
IJCAI 2026, Bremen, Germany, 15-21 August 2026. ijcai.org, 2026.

## Introduction
Retrieval-Augmented Generation (RAG) is sensitive to the vast hyperparameters of the retriever and generator, yet optimizing them using given queries remains a challenging task due to the complex hyperparameter interactions and expensive evaluation costs. Existing algorithms are ineffective and slow in convergence, since they often treat RAG as a monolithic black box or only optimize partial hyperparameters. In this paper, we propose CDS4RAG, a framework that optimizes the full hyperparameters of RAG using given queries via a new formulation of cyclic dual-sequential problem. CDS4RAG is special in the sense that it distinguishes the hyperparameters of the retriever and generator, optimizing them in turn and in a cyclic manner. Such a paradigm allows us to design fine-grained within-cycle budget provision and expedite the optimization via cross-cycle seeding when optimizing the generator. Importantly, CDS4RAG is an algorithm-agnostic framework that can be paired with diverse general algorithms. Through experiments on four common benchmarks and two backbone LLMs, we reveal that CDS4RAG considerably boosts the vanilla algorithms in 21/24 cases while significantly outperforming state-of-the-art algorithms in all cases with up to 1.54X improvements of generation quality and better speedup.

## Code Structure
   - darasets => There is one of our four benchmarks, others are omited due to the file size restricted by github and will be accessed later <br>
   - util/utils => Essential util for CDS4RAG <br>
   - Run_util => Core evaluation fuction <br>
   - requirements.txt => Essential requirments need to be installed <br>
   - CDS4RAG.py => The reproduction code of CDS4RAG and is equiped with the overall best tuner HEBO
   - raw_results => Experimental outputs organized by backbone model, optimization method, and benchmark dataset

##  <a name='quick-start'></a> Quick Start

* Python 3.9+

To run the code, and install the essential requirements: 
```
pip install -r requirements.txt
```
Then you need to force the installation and ignore error prompts — although there is a numpy incompatibility issue, it does not actually affect the runtime:
```
pip install HEBO==0.3.6
```
 - Note: Before running the code, you need to first install Ollama locally which can be refered in https://github.com/ollama/ollama Then pull the models llama3.1:8b (qwen3:8b) and bge-m3:567m, and grant the necessary permissions for them.

And you can run the below code to have a quick start:
```
python3 CDS4RAG.py
```
Next, the code will run with llama-3.1-7B on the agriculture bencmark.


## Parameter Space

The core parameter space of RAG that we currently encompass; essentially, our method can additionally incorporate extra parameters, such as prompt templates, rerank thresholds, and so on.
| Phase | Hyperparameter | Type | Range / Values |
|-------|---------------|------|----------------|
| Retriever ($\Phi$) | Database Choice | Categorical | \{DuckDB, Chroma, FAISS\} |
| | Chunk Size | Integer | [256, 1024] |
| | Chunk Overlap | Integer | [32, 128] |
| | Embedding Temperature | Float | [0.0, 1.0] |
| | Embedding Window | Integer | [512, 2048] |
| | Embedding Repeat Penalty | Float | [0.9, 1.5] |
| | Embedding Top-$k$ | Integer | [10, 100] |
| Generator ($\Theta$) | Retrieval Numbers ($K$) | Integer | [1, 10] |
| | Generation Temperature | Float | [0.0, 1.0] |
| | Generation Window | Integer | [512, 8192] |
| | Generation Repeat Penalty | Float | [0.9, 1.5] |
| | Generation Top-$k$ | Integer | [10, 100] |
