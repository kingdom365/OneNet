import ollama

class Retriever:
    def __init__(self, client, emb_model="chroma/all-minilm-l6-v2-f32"):
        self.emb_model = emb_model
        self.client = client    # chromadb persistent-client
    
    # 在外部KB（向量数据库）中搜索k近邻实体
    def get_candidates(self, mention, k, collection):
        # 1. create mention embedding
        mention_surform = mention['label']
        mentino_desc = mention['definition']
        mention_pr = mention_surform + ":" + mentino_desc
        ollama_resp = ollama.embeddings(model=self.emb_model, prompt=mention_pr)
        mention_emb = ollama_resp['embedding']
        
        # 2. k-nn search (based on hnsw)
        rs = self.client.get_or_create_collection(
            collection
        )
        print('result set size : ', rs.count())
        docs = rs.query(
            query_embeddings=[mention_emb],
            n_results=k
        )
        print('candidates get num : ', len(docs['ids'][0]))
        # {ids, embeddings, documents, distances}
        # print(docs)
        # 3. return entities
        return docs["documents"][0]
    