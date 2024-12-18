"""
    RAG module
    将所有的entity summary存储到向量数据库
"""
from chromadb import Settings
import ollama
import chromadb

class Indexer:
    # 将(entity_name, entity_summary)注入数据库
    def __init__(self, client, emb_model=None, db_path=None, metadata=None):
        if emb_model is None:
            self.emb_model = 'chroma/all-minilm-l6-v2-f32' # 基于ollama，默认
        else:
            self.emb_model = emb_model

        if db_path is None:
            self.db_path = './chroma.db'
        else:
            self.db_path = db_path

        if metadata is None:
            self.metadata = {"hnsw:space": "cosine"}
        else:
            self.metadata = metadata

        self.collection_name = 'general'

        # chromadb client
        self.client = client

    # 添加实体信息；将(doc_id, entity_name, summary)写入db
    def insert_entity(self, doc_id, entity_name, entity_summary):
        collection = self.collection_name
        try:
            rs = self.client.get_collection(collection)
        except Exception as e:
            print("Indexer: no such collection, creating...")
            rs = self.client.create_collection(name=collection)
        entity_info = entity_name + ":" + entity_summary
        ollama_resp = ollama.embeddings(model=self.emb_model, prompt=entity_info)
        embedding = ollama_resp['embedding']    # 实体信息嵌入
        try:
        # 写入实体信息
            rs.add(
                ids=[doc_id],
                documents=[entity_info],
                embeddings=[embedding],
                metadatas=[self.metadata]
            )
            print("insert entity {} successfully".format(entity_name))
            return True
        except Exception as e: 
            print("insert entity {} failed!".format(entity_name))
            print("Reason : {}".format(e))
            return False

    # 刷新数据库
    def refresh_and_get_db(self):
        try:
            collection = self.client.get_collection(self.collection_name)
            if collection.count() > 0:
                return collection.count(), collection
            self.client.delete_collection(self.collection_name)
            collection = self.client.create_collection(self.collection_name)
        except Exception as e:
            print("collection {} does not exist, creating...".format(self.collection_name))
            collection = self.client.create_collection(self.collection_name)
        return 0, collection
    

