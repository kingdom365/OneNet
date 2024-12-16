"""
    NER阶段，包含NER和desc生成
"""
from LLM_calls import llm_call

DESCRIB_PROMPT = 'Given a text and a semicolon-separated list of entities from that text, write a definition for each entity in the following format:\n'\
'entity: <A single sentence definition that explains the meaning of the entity>'

class NER:
    def __init__(self, pipline, llm_name='Qwen'):
        self.pipline = pipline  # 包含llm的tokenizer和llm本身
        self.llm_name = llm_name

    # 根据目标类实体（一个笼统的概括，比如军事装备的舰艇、装甲车、步枪等）从input_text中获取mentions
    # 再给每个mention生成desc，达到与db中的语义一致性
    def extract_mentions(self, input_text, fields):
        system_prompt = 'From the text below, extract all mentions of the following entities in the following format:\n\n' 
        system_prompt += fields[0] + ' It must be semicolon-separated. If no mention of ' + fields[0].split(":")[0] + ' is found in text, respond None.>'
        system_prompt += "\n\nText:\n"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": input_text}
        ]

        response = llm_call(messages=messages, model_name=self.llm_name, pipeline=self.pipline)
        mentions = response.split(":", -1)[-1].strip().split(";")
        mentions = [m.strip() for m in mentions if m.strip() and "none" not in m.lower()] 

        user_content = f'\nHere is the text:\n{input_text}\nHere is the list of entities:\n{", ".join(mentions)}'
        desc_msg = [
            {"role": "system", "content": DESCRIB_PROMPT},
            {"role": "user", "content": user_content},
        ]
        descs_response = llm_call(messages=desc_msg, model_name=self.llm_name, pipeline=self.pipline)
        
        parsed_md = []
        for desc in descs_response.split("\n"):
            if ":" in desc.strip():
                tokens = desc.strip().split(":")
                label, definition = tokens[0].strip(), tokens[1].strip()
                if len(tokens) > 2:
                    print("Warning: length of (m,d) more than 2")
                parsed_md.append({"label": label, "definition": definition})
        # list of (m, d)
        return parsed_md