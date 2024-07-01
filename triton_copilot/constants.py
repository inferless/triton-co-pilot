system_prompt_config_pbtxt = """
                Your task is to generate config.pbtxt for the given code snippet in correct format for nvidia triton server.
                User may also provide input.json, output.json and input_schema.py. Infer the necessary information from these files.
                Below are two examples for the code snippet
                ## Example 1
                from threading import Thread
                from typing import Iterator
                import os

                import torch
                import json
                from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

                model_id = "meta-llama/Llama-2-7b-chat-hf"


                class PythonModel:
                    def get_prompt(self, message, chat_history, system_prompt):
                        texts = [f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n"]
                        chat_history = json.loads(chat_history)

                        for each in chat_history:
                            print("each", each, flush=True)
                            user_input = each["user_input"]
                            response = each["response"]
                            texts.append(
                                f"{user_input.strip()} [/INST] {response.strip()} </s><s> [INST] "
                            )
                        texts.append(f"{message.strip()} [/INST]")
                        return "".join(texts)

                    def get_input_token_length(self, message, chat_history, system_prompt):
                        prompt = self.get_prompt(message, chat_history, system_prompt)
                        input_ids = self.tokenizer([prompt], return_tensors="np")["input_ids"]
                        return input_ids.shape[-1]

                    def run_function(
                        self,
                        message,
                        chat_history,
                        system_prompt,
                        max_new_tokens=1024,
                        temperature=0.8,
                        top_p=0.95,
                        top_k=5,
                    ):
                        prompt = self.get_prompt(message, chat_history, system_prompt)
                        inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda")

                        streamer = TextIteratorStreamer(
                            self.tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True
                        )
                        generate_kwargs = dict(
                            inputs,
                            streamer=streamer,
                            max_new_tokens=max_new_tokens,
                            do_sample=True,
                            top_p=top_p,
                            top_k=top_k,
                            temperature=temperature,
                            num_beams=1,
                        )
                        t = Thread(target=self.model.generate, kwargs=generate_kwargs)
                        t.start()

                        outputs = ""
                        for text in streamer:
                            outputs += text

                        return outputs

                    def load(self):
                        token = "abscafs"
                        self.tokenizer = AutoTokenizer.from_pretrained(
                            model_id, use_auth_token=token, device="cuda"
                        )
                        if torch.cuda.is_available():
                            self.model = AutoModelForCausalLM.from_pretrained(
                                model_id,
                                torch_dtype=torch.float16,
                                device_map="auto",
                                use_auth_token=token,
                            )
                        else:
                            self.model = None

                    def infer(self, inputs):
                        message = inputs["message"]
                        chat_history = inputs["chat_history"] if "chat_history" in inputs else []
                        system_prompt = inputs["system_prompt"] if "system_prompt" in inputs else ""
                        result = self.run_function(
                            message=message,
                            chat_history=chat_history,
                            system_prompt=system_prompt,
                        )
                        return {"generated_text": result}

                    def unload(self):
                        self.tokenizer = None
                        self.model = None


                config.pbtxt

                name: "new_llama2_7b_chat_0b2812185e684a56b7a9f15b68cbe6ee"
                backend: "python"
                input [
                    {
                        name: "message", 
                        dims: [1], 
                        data_type: TYPE_STRING
                    }, 
                    {
                        name: "chat_history", 
                        dims: [1], 
                        data_type: TYPE_STRING
                    }, 
                    {
                        name: "system_prompt", 
                        dims: [1], 
                        data_type: TYPE_STRING
                    }
                ]
                output [
                    {
                        name: "generated_text", 
                        dims: [1], 
                        data_type: TYPE_STRING
                    }
                ]
                instance_group [
                    {
                        kind: KIND_GPU
                    }
                ]


                ## Example 2
                import os

                os.environ[
                    "TRANSFORMERS_CACHE"
                ] = "/opt/tritonserver/model_repository/falcon7b/hf_cache"
                import json

                import numpy as np
                import torch
                import transformers
                import triton_python_backend_utils as pb_utils


                class TritonPythonModel:
                    def initialize(self, args):
                        self.logger = pb_utils.Logger
                        self.model_config = json.loads(args["model_config"])
                        self.model_params = self.model_config.get("parameters", {})
                        default_hf_model = "tiiuae/falcon-7b"
                        default_max_gen_length = "15"
                        # Check for user-specified model name in model config parameters
                        hf_model = self.model_params.get("huggingface_model", {}).get(
                            "string_value", default_hf_model
                        )
                        # Check for user-specified max length in model config parameters
                        self.max_output_length = int(
                            self.model_params.get("max_output_length", {}).get(
                                "string_value", default_max_gen_length
                            )
                        )

                        self.logger.log_info(f"Max sequence length: {self.max_output_length}")
                        self.logger.log_info(f"Loading HuggingFace model: {hf_model}...")
                        # Assume tokenizer available for same model
                        self.tokenizer = transformers.AutoTokenizer.from_pretrained(hf_model)
                        self.pipeline = transformers.pipeline(
                            "text-generation",
                            model=hf_model,
                            torch_dtype=torch.float16,
                            tokenizer=self.tokenizer,
                            device_map="auto",
                        )
                        self.pipeline.tokenizer.pad_token_id = self.tokenizer.eos_token_id

                    def execute(self, requests):
                        prompts = []
                        for request in requests:
                            input_tensor = pb_utils.get_input_tensor_by_name(request, "text_input")
                            multi_dim = input_tensor.as_numpy().ndim > 1
                            if not multi_dim:
                                prompt = input_tensor.as_numpy()[0].decode("utf-8")
                                self.logger.log_info(f"Generating sequences for text_input: {prompt}")
                                prompts.append(prompt)
                            else:
                                # Implementation to accept dynamically batched inputs
                                num_prompts = input_tensor.as_numpy().shape[0]
                                for prompt_index in range(0, num_prompts):
                                    prompt = input_tensor.as_numpy()[prompt_index][0].decode("utf-8")
                                    prompts.append(prompt)

                        batch_size = len(prompts)
                        return self.generate(prompts, batch_size)

                    def generate(self, prompts, batch_size):
                        sequences = self.pipeline(
                            prompts,
                            max_length=self.max_output_length,
                            pad_token_id=self.tokenizer.eos_token_id,
                            batch_size=batch_size,
                        )
                        responses = []
                        texts = []
                        for i, seq in enumerate(sequences):
                            output_tensors = []
                            text = seq[0]["generated_text"]
                            texts.append(text)
                            tensor = pb_utils.Tensor("text_output", np.array(texts, dtype=np.object_))
                            output_tensors.append(tensor)
                            responses.append(pb_utils.InferenceResponse(output_tensors=output_tensors))

                        return responses

                    def finalize(self):
                        print("Cleaning up...")

                config.pbtxt
                
                name: "falcon-7basdfvdawreq7a9f15b68cbe6ee"
                backend: "python"
                parameters: {
                    key: "huggingface_model",
                    value: {string_value: "tiiuae/falcon-7b"}
                }
                parameters: {
                    key: "max_output_length",
                    value: {string_value: "15"}
                }
                input [
                    {
                        name: "text_input"
                        data_type: TYPE_STRING
                        dims: [ 1 ]
                    }
                ]
                output [
                    {
                        name: "text_output"
                        data_type: TYPE_STRING
                        dims: [ -1 ]
                    }
                ]
                instance_group [
                    {
                        kind: KIND_GPU
                    }
                ]
            
            In your response, Just provide the code snippet without any quotes or comments
            """
