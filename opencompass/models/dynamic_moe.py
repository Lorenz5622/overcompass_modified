

class HuggingFaceDynamicMoE(HuggingFace):

    def __init__(
        self,
        path: str,
        hf_cache_dir: Optional[str] = None,
        max_seq_len: int = 2048,
        tokenizer_path: Optional[str] = None,
        tokenizer_kwargs: dict = dict(),
        peft_path: Optional[str] = None,
        tokenizer_only: bool = False,
        model_kwargs: dict = dict(device_map="auto"),
        generation_kwargs: dict = dict(),
        meta_template: Optional[Dict] = None,
        extract_pred_after_decode: bool = False,
        batch_padding: bool = False,
        pad_token_id: Optional[int] = None,
        mode: str = "none",
        num_extra_tokens: int = 50,
    ):
        super().__init__(
            path=path,
            hf_cache_dir=hf_cache_dir,
            max_seq_len=max_seq_len,
            tokenizer_path=tokenizer_path,
            tokenizer_kwargs=tokenizer_kwargs,
            peft_path=peft_path,
            tokenizer_only=tokenizer_only,
            generation_kwargs=generation_kwargs,
            model_kwargs=model_kwargs,
            meta_template=meta_template,
            extract_pred_after_decode=extract_pred_after_decode,
            batch_padding=batch_padding,
            pad_token_id=pad_token_id,
            mode=mode,
        )

        from transformers import LlamaTokenizer
        # from Dynamic_MoE.modeling.modeling_moe_ori import MoEForCausalLM
        from Dynamic_MoE.modeling.modeling_moe import MoEForCausalLM
        from Dynamic_MoE.modeling.configuration_moe import MoEConfig

        self.tokenizer = LlamaTokenizer.from_pretrained(path)
        self.tokenizer.pad_token = self.tokenizer.unk_token

        model_config = MoEConfig.from_pretrained(path, trust_remote_code=True)
        self.model = MoEForCausalLM.from_pretrained(
            path,
            from_tf=False,
            config=model_config,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        ).cuda()
        self.model.eval()

    def generate(
        self,
        inputs: List[PromptType],
        max_out_len: int = 512,
        **kwargs,
    ) -> str:
        tokens = self.tokenizer(inputs, return_tensors="pt")
        input_ids = tokens.input_ids.cuda()
        print("generate")
        generate_ids = self.model.generate(
            inputs=input_ids,
            num_beams=1,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            max_new_tokens=max_out_len,
            top_p=0.9,
            temperature=1.0,
            do_sample=True,
            dynamic_k=[2,0,6,7,3,6,4,7,7,4,1,3,6,2,7,5,5,4,1,4,2,5,1,4,2,7,7,7,1,0,5,7],
        )
        outputs = self.tokenizer.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        response = [outputs[i][len(inputs[i]) :] for i in range(len(outputs))][0]
        return response
    
    def get_token_len(self, prompt: str) -> int:
        return len(self.tokenizer.encode(prompt))

    
