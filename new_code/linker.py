from LLM_calls import load_llm, llm_call


class Linker:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.model_name = 'Qwen'
    
    # 对于给定的mention，LLM生成(m, d)
    def describe_prompt(self, mention, lctx, rctx):
        content = "Given a text and an entity from that text, write a definition for the entity in the following format:\n\n"    
        content += "<entity>:<definition content> \n\n"
        text = lctx + " " + mention + " " + rctx
        content += "Here is the text:\n{}".format(text) 
        content += "\nHere is the entity:\n{}".format(mention)
        return content

    def mention_desc_generate(self, prompt_content):
        msgs = [
            {"role": "system", "content": ""},
            {"role": "user", "content": prompt_content}
        ]
        response = llm_call(msgs, self.model_name, pipeline=self.pipeline)
        return response.split(':')[-1]

    # 输入(m, ctx)，cand_list(cand, desc)
    # 输出(cand, desc, yes/no)表示是否被链接到
    def pointwise_prompt(self, mention, lctx, rctx, cand, instruction_dict):
        content = "You're an entity disambiguator. I'll give you the description of entity disambiguation and some tips on entity disambiguation, and you need to pay attention to these textual features: \n\n" 
        content += instruction_dict[0]['prompt']
        content += "\n\n"
        content += "Now, I'll give you a mention, a context, and a candidate entity, and the mention will be highlighted with '###'.\n\n"
        content += "Mention:{}\n".format(mention)

        ctx = lctx + " ###" + mention + "### " + rctx
        ctx.strip()
        ctx = ' '.join(ctx.split())
        content += "Context:{}\n".format(ctx)

        candidate_entity = "{}:{}".format(cand['entity_name'], cand['entity_desc'])
        content += 'candidate entity:{}\n\n'.format(candidate_entity)

        content += """You need to determine if the mention and the candidate entity are related. Please refer to the above tips and give your reasons, and finally answer 'yes' or 'no'. Answer 'yes' when you think the information is insufficient or uncertain."""
        return content


    # 输入(m, ctx), (cand, desc, yes)
    # 输出最终确定的链接实体
    def listwise_prompt(self, mention, lctx, rctx, cands, instruction_dict, prompt_id=1):
        system_content = "You're an entity disambiguator. I'll give you the description of entity disambiguation and some tips on entity disambiguation, you should pay attention to these textual features:\n\n"
        system_content += instruction_dict[prompt_id]['prompt']
        
        # mention
        content = "Now, I'll give you a mention, a context, and a list of candidates entities, the mention will be highlighted with '###' in context.\n\n"
        content += "Mention:{}\n".format(mention)
        for i, cand in enumerate(cands):
            candidate_entity = "{}.{}".format(cand['entity_name'], cand['entity_desc'])
            content += "Entity {}:{}".format(i + 1, candidate_entity)
        content += "\n"
        content += "You need to determine which candidate entity is more likely to be the mention. Please refer to the above example, give your reasons, and finally answer serial number of the entity and the name of the entity. If all candidate entities are not appropriate, you can answer '-1.None'."
        return system_content, content

    # 输入pointwise-pr，输出结果
    def point_wise_el(self, prompt_content, cand):
        msgs = [
            {"role": "system", "content": ""},
            {"role": "user", "content": prompt_content}
        ]
        response = llm_call(msgs, self.model_name, pipeline=self.pipeline)
        if 'yes' in response:
            return cand, True
        else:
            return cand, False

    # 输入listwise-pr，输出结果
    def list_wise_el(self, system_content, user_content):
        msgs = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]
        response = llm_call(msgs, self.model_name, pipeline=self.pipeline)
        return response