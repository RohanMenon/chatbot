# This file contains the ModelInterface class that is responsible 
# for loading the model and generating responses to input text. 

from transformers import AutoTokenizer, AutoModelForCausalLM
from time import time
from download_model import download_model

class ModelInterface(object):
    """Model interface for chatbot."""

    def __init__(self):
        self.system_message = """
            You are an AI agent tasked to answer general questions in 
            a simple and short way.    
    """
        self.path_to_model = download_model()
        self.max_new_tokens = 128
        self.initialize_model()

    def initialize_model(self) -> None:
        """"Load tokenizer and LM model."""
        start_time = time()
        self.tokenizer = AutoTokenizer.from_pretrained(self.path_to_model)
        tok_time = time()
        print(f"Load tokenizer: {round(tok_time-start_time, 1)} sec.")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.path_to_model,
            return_dict=True,
            low_cpu_mem_usage=True,
            device_map="auto",
            trust_remote_code=True
        )
        mod_time = time()
        print(f"Load model: {round(mod_time-tok_time, 1)} sec.")

    @staticmethod
    def clean_answer(answer, input_text) -> str:
        """Remove unwanted tokens from input text."""
        answer = answer.replace(input_text,"")
        answer = answer.replace("<end_of_turn>", "")
        answer = answer.replace("<bos>", "")
        answer = answer.replace("<eos>", "")
        answer = answer.lstrip()
        return answer


    def get_message_response(self, input_text) -> dict:
        """Return model response for a given input text."""
        start_time = time()

        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        input_ids = self.tokenizer(input_text, return_tensors="pt").to("mps")

        outputs = self.model.generate(
                **input_ids,
                do_sample=True,
                top_k=10,
                temperature=0.1,
                top_p=0.95,
                num_return_sequences=1,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=terminators[0]
                )
        end_time = time()
        answer = self.clean_answer(f"{self.tokenizer.decode(outputs[0])}",
                                   input_text)

        print(f"Total response time: {round(end_time-start_time, 1)} sec.")


        return {
                "input": input_text,
                "response": answer,
                "response_time":  f"{round(end_time-start_time, 1)} sec."
                }
        