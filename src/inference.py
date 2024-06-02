from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import torch

class LLMInference():
    
    def __init__(self, model_loc: str):
        print("initializing inference")
        self.model_loc = model_loc
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(model_loc, quantization_config=bnb_config)
        self.tokenizer = AutoTokenizer.from_pretrained(model_loc)

        print("setting up pipeline")
        self.llm_pipeline = pipeline(
            model=model,
            tokenizer=self.tokenizer,
            task="text-generation",
            do_sample=True,
            temperature=0.2,
            repetition_penalty=1.1,
            return_full_text=False,
            max_new_tokens=500,
        )
    
    def set_rag_prompt(self):
        rag_prompt_in_chat_format = [
            {
                "role": "system",
                "content": """Using the information contained in the context,
        give a comprehensive answer to the question.
        Respond only to the question asked, response should be concise and relevant to the question.
        Provide the number of the source document when relevant.
        If the answer cannot be deduced from the context, do not give an answer.""",
            },
            {
                "role": "user",
                "content": """Context:
        {context}
        ---
        Now here is the question you need to answer.

        Question: {question}""",
            },
        ]
        self.rag_prompt_template = self.tokenizer.apply_chat_template(
            rag_prompt_in_chat_format, tokenize=False, add_generation_prompt=True
        )
    
    def no_rag_prompt(self):
        llm_prompt_in_chat_format = [
            {
                "role": "system",
                "content": """Give a comprehensive answer to the question.
        Respond only to the question asked, response should be concise and relevant to the question.
        Provide the source when relevant.
        If the answer cannot be deduced from the context, do not give an answer.""",
            },
            {
                "role": "user",
                "content": """
        Here is the question you need to answer.

        Question: {question}""",
            },
        ]
        return self.tokenizer.apply_chat_template(
            llm_prompt_in_chat_format, tokenize=False, add_generation_prompt=True
        )