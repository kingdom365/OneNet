from ner import NER
from retriever import Retriever
from linker import Linker
from indexer import Indexer
import json
import chromadb
from chromadb import Settings
from LLM_calls import load_llm
from linker import Linker
from prompt import read_prompt

summary_path = "../datasets/zeshel/process_data/summary.json"
dataset_path = "../datasets/aida/aida_test.jsonl"
prompts_path = "../prompt/prompt.jsonl"

vdb_path = "../chroma.db"
chromadb_client = chromadb.PersistentClient(
            path=vdb_path,
            settings=Settings(allow_reset=False, anonymized_telemetry=False)
        )

pipeline = load_llm('Qwen', '../Qwen/Qwen2.5-1.5B-Instruct')

def init_db():
    # 将summary pr得到的entity summary注入vdb
    indexer = Indexer(client=chromadb_client)
    summary = []
    with open(summary_path, 'r') as f:
        for line in f:
            summary.append(json.loads(line.strip()))
    sz, collection = indexer.refresh_and_get_db()
    error_items = []
    if sz <= 0:
        for _, ent in enumerate(summary):
            doc_id = ent['document_id'] 
            desc = ent['summary']
            name = ent['title']
            ret = indexer.insert_entity(doc_id, name, desc)
            if not ret:
                error_items.append({"doc_id": doc_id, "desc": desc, "name": name})
    return 'general'


if __name__ == '__main__':
    # NER得到若干实体
    # 前序获取summary，这里注入vdb
    collection = init_db()

    # 1. 加载测试集
    ner_results = []
    with open(dataset_path, 'r') as f:
        for line in f:
            ner_results.append(json.loads(line.strip()))

    # 2. 提取所有mention，搜索top-10近邻的数据
    retriever_instance = Retriever(client=chromadb_client)
    linker_instance = Linker(pipeline=pipeline)
    instruction_pr = read_prompt(prompts_path)
    mentions = []
    for r in ner_results:
        mention = r['mention']
        lctx = r['left_context']
        rctx = r['right_context']
        print('mention : ', mention)

        # 构造(m, d)
        print("******** describe prompt ***********")
        mention_desc_pr = linker_instance.describe_prompt(mention, lctx, rctx)
        print("          ********** generate desc ***********")
        definition = linker_instance.mention_desc_generate(mention_desc_pr)
        print("          ********** (m, d) ***********")
        print(mention, ":", definition)
        label = mention
        # definition = lctx + " " + mention + " " + rctx
        print("******** get candidates ***********")
        cands = retriever_instance.get_candidates({"label": label, "definition": definition}, 10, collection)
        # 3.1 执行point-wise el
        linkable_cands = []
        for cand in cands:
            entity_name = cand.split(":")[0]
            entity_desc = cand.split(":")[-1]
            print('cand : ', entity_name)
            sys_pr, user_pr = linker_instance.pointwise_prompt(mention, lctx, rctx, {"entity_name": entity_name, "entity_desc": entity_desc}, instruction_pr) 
            res_cand, res = linker_instance.point_wise_el(sys_pr, user_pr, entity_name)
            print('{}---{}: {}'.format(mention, entity_name, res))
            if res:
                # link
                linkable_cands.append({"entity_name": entity_name, "entity_desc": entity_desc})
        if len(linkable_cands) > 0:
            # 3.2 执行list-wise el
            print("list-wise el")
            system_content, user_content = linker_instance.listwise_prompt(mention, lctx, rctx, linkable_cands, instruction_pr)
            res = linker_instance.list_wise_el(system_content, user_content)
            print("{}--{}\n".format(mention, res))
        else:
            print("no candidates found")

    