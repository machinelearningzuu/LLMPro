import yaml, os, camelot
from typing import List, Dict
from llama_index.schema import IndexNode
from llama_index.llms import AzureOpenAI
from llama_index.llm_predictor import LLMPredictor
from llama_index import set_global_service_context
from llama_index.node_parser import SimpleNodeParser
from llama_index.retrievers import RecursiveRetriever
from llama_hub.file.pymu_pdf.base import PyMuPDFReader
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.response_synthesizers import get_response_synthesizer
from llama_index.query_engine import PandasQueryEngine, RetrieverQueryEngine
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext

with open('/Users/1zuu/Desktop/LLM RESEARCH/LLMPro/cadentials.yaml') as f:
    credentials = yaml.load(f, Loader=yaml.FullLoader)

os.environ['AD_OPENAI_API_KEY'] = credentials['AD_OPENAI_API_KEY']

embedding_llm = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
llm=AzureOpenAI(
                deployment_name=credentials['AD_DEPLOYMENT_ID'],
                model=credentials['AD_ENGINE'],
                api_key=credentials['AD_OPENAI_API_KEY'],
                api_version=credentials['AD_OPENAI_API_VERSION'],
                azure_endpoint=credentials['AD_OPENAI_API_BASE']
                )
chat_llm = LLMPredictor(llm)

service_context = ServiceContext.from_defaults(
                                                embed_model=embedding_llm,
                                                llm_predictor=chat_llm
                                                )
set_global_service_context(service_context)

def get_tables(path: str, pages: List[int]):
    table_dfs = []
    for page in pages:
        table_list = camelot.read_pdf(path, pages=str(page))
        table_df = table_list[0].df
        table_df = (
            table_df.rename(columns=table_df.iloc[0])
            .drop(table_df.index[0])
            .reset_index(drop=True)
        )
        table_dfs.append(table_df)
    return table_dfs

def get_texts(path: str, pages: List[int]):
    reader = PyMuPDFReader() 
    docs = reader.load(path)
    return docs

table_dfs = get_tables('./data/billionaires_page.pdf', pages=[3, 25])
df_query_engines = [
                    PandasQueryEngine(table_df, service_context=service_context)
                    for table_df in table_dfs
                    ]
summaries = [
            (
                "This node provides information about the world's richest billionaires"
                " in 2023"
            ),
            (
                "This node provides information on the number of billionaires and"
                " their combined net worth from 2000 to 2023."
            )
            ]

df_nodes = [
            IndexNode(text=summary, index_id=f"pandas{idx}")
            for idx, summary in enumerate(summaries)
            ]

df_id_query_engine_mapping = {
                            f"pandas{idx}": df_query_engine
                            for idx, df_query_engine in enumerate(df_query_engines)
                        }
docs = get_texts('./data/billionaires_page.pdf', pages=[3, 25])
doc_nodes = service_context.node_parser.get_nodes_from_documents(docs)

vector_index_text = VectorStoreIndex(doc_nodes)
vector_index = VectorStoreIndex(doc_nodes + df_nodes)

vector_index_text.storage_context.persist(persist_dir="./db/text_index")
vector_index.storage_context.persist(persist_dir="./db/recursive_index")