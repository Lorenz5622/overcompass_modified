import csv
import math
import os
import re
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import transformers

from opencompass.models.base import BaseModel
from opencompass.models.base_api import APITemplateParser
from opencompass.registry import MODELS
from opencompass.utils.logging import get_logger
from opencompass.utils.prompt import PromptList

PromptType = Union[PromptList, str]


class MultiTokenEOSCriteria(transformers.StoppingCriteria):
    """Criteria to stop on the specified multi-token sequence."""

    def __init__(
        self,
        sequence: str,
        tokenizer: transformers.PreTrainedTokenizer,
        batch_size: int,
    ):
        self.done_tracker = [False] * batch_size
        self.sequence = sequence
        self.sequence_ids = tokenizer.encode(sequence,
                                             add_special_tokens=False)
        self.sequence_id_len = len(self.sequence_ids)
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        # compare the last len(stop) tokens
        lookback_ids_batch = input_ids[:, -self.sequence_id_len:]
        lookback_tokens_batch = self.tokenizer.batch_decode(lookback_ids_batch)
        for i, done in enumerate(self.done_tracker):
            if done:
                continue
            self.done_tracker[i] = self.sequence in lookback_tokens_batch[i]
        return False not in self.done_tracker


@MODELS.register_module()
class HuggingFace(BaseModel):
    """Model wrapper around HuggingFace models.

    Args:
        path (str): The name or path to HuggingFace's model.
        hf_cache_dir: Set the cache dir to HF model cache dir. If None, it will
            use the env variable HF_MODEL_HUB. Defaults to None.
        max_seq_len (int): The maximum length of the input sequence. Defaults
            to 2048.
        tokenizer_path (str): The path to the tokenizer. Defaults to None.
        tokenizer_kwargs (dict): Keyword arguments for the tokenizer.
            Defaults to {}.
        peft_path (str, optional): The name or path to the HuggingFace's PEFT
            model. If None, the original model will not be converted to PEFT.
            Defaults to None.
        tokenizer_only (bool): If True, only the tokenizer will be initialized.
            Defaults to False.
        model_kwargs (dict): Keyword arguments for the model, used in loader.
            Defaults to dict(device_map='auto').
        meta_template (Dict, optional): The model's meta prompt
            template if needed, in case the requirement of injecting or
            wrapping of any meta instructions.
        extract_pred_after_decode (bool): Whether to extract the prediction
            string from the decoded output string, instead of extract the
            prediction tokens before decoding. Defaults to False.
        batch_padding (bool): If False, inference with be performed in for-loop
            without batch padding.
        pad_token_id (int): The id of the padding token. Defaults to None. Use
            (#vocab + pad_token_id) if get negative value.
        mode (str, optional): The method of input truncation when input length
            exceeds max_seq_len. 'mid' represents the part of input to
            truncate. Defaults to 'none'.
        use_fastchat_template (str, optional): Whether to use fastchat to get
            the conversation template. If True, fastchat needs to be
            implemented first. Defaults to False.
        end_str (str, optional): Whether to trim generated strings with end_str
            if the model has special ending strings that are not handled well.
            Defaults to None.

    Note:
        About ``extract_pred_after_decode``: Commonly, we should extract the
        the prediction tokens before decoding. But for some tokenizers using
        ``sentencepiece``, like LLaMA,  this behavior may change the number of
        whitespaces, which is harmful for Python programming tasks.
    """

    def __init__(self,
                 path: str,
                 hf_cache_dir: Optional[str] = None,
                 max_seq_len: int = 2048,
                 tokenizer_path: Optional[str] = None,
                 tokenizer_kwargs: dict = dict(),
                 peft_path: Optional[str] = None,
                 tokenizer_only: bool = False,
                 model_kwargs: dict = dict(device_map='auto'),
                 generation_kwargs: dict = dict(),
                 meta_template: Optional[Dict] = None,
                 extract_pred_after_decode: bool = False,
                 batch_padding: bool = False,
                 pad_token_id: Optional[int] = None,
                 mode: str = 'none',
                 use_fastchat_template: bool = False,
                 end_str: Optional[str] = None):
        super().__init__(path=path,
                         max_seq_len=max_seq_len,
                         tokenizer_only=tokenizer_only,
                         meta_template=meta_template)
        if hf_cache_dir is None:
            hf_cache_dir = os.getenv('HF_MODEL_HUB', None)
        self.logger = get_logger()
        self.pad_token_id = pad_token_id
        assert mode in ['none', 'mid']
        self.mode = mode
        self._load_tokenizer(path=path,
                             tokenizer_path=tokenizer_path,
                             tokenizer_kwargs=tokenizer_kwargs)
        self.batch_padding = batch_padding
        self.extract_pred_after_decode = extract_pred_after_decode
        if not tokenizer_only:
            self._load_model(path=path,
                             model_kwargs=model_kwargs,
                             peft_path=peft_path)
        self.generation_kwargs = generation_kwargs
        self.use_fastchat_template = use_fastchat_template
        self.end_str = end_str

    def _load_tokenizer(self, path: str, tokenizer_path: Optional[str],
                        tokenizer_kwargs: dict):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path if tokenizer_path else path, **tokenizer_kwargs)

        # A patch for some models without pad_token_id
        if self.pad_token_id is not None:
            if self.pad_token_id < 0:
                self.pad_token_id += self.tokenizer.vocab_size
            if self.tokenizer.pad_token_id is None:
                self.logger.debug(f'Using {self.pad_token_id} as pad_token_id')
            elif self.tokenizer.pad_token_id != self.pad_token_id:
                self.logger.warning(
                    'pad_token_id is not consistent with the tokenizer. Using '
                    f'{self.pad_token_id} as pad_token_id')
            self.tokenizer.pad_token_id = self.pad_token_id
        elif self.tokenizer.pad_token_id is None:
            self.logger.warning('pad_token_id is not set for the tokenizer.')
            if self.tokenizer.eos_token is not None:
                self.logger.warning(
                    f'Using eos_token_id {self.tokenizer.eos_token} '
                    'as pad_token_id.')
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                from transformers.generation import GenerationConfig
                gcfg = GenerationConfig.from_pretrained(path)

                if gcfg.pad_token_id is not None:
                    self.logger.warning(
                        f'Using pad_token_id {gcfg.pad_token_id} '
                        'as pad_token_id.')
                    self.tokenizer.pad_token_id = gcfg.pad_token_id
                else:
                    raise ValueError(
                        'pad_token_id is not set for this tokenizer. Try to '
                        'set pad_token_id via passing '
                        '`pad_token_id={PAD_TOKEN_ID}` in model_cfg.')

        # A patch for llama when batch_padding = True
        if 'decapoda-research/llama' in path or \
                (tokenizer_path and
                 'decapoda-research/llama' in tokenizer_path):
            self.logger.warning('We set new pad_token_id for LLaMA model')
            # keep consistent with official LLaMA repo
            # https://github.com/google/sentencepiece/blob/master/python/sentencepiece_python_module_example.ipynb  # noqa
            self.tokenizer.bos_token = '<s>'
            self.tokenizer.eos_token = '</s>'
            self.tokenizer.pad_token_id = 0

    def _set_model_kwargs_torch_dtype(self, model_kwargs):
        if 'torch_dtype' not in model_kwargs:
            torch_dtype = torch.float16
        else:
            torch_dtype = {
                'torch.float16': torch.float16,
                'torch.bfloat16': torch.bfloat16,
                'torch.float': torch.float,
                'auto': 'auto',
                'None': None
            }.get(model_kwargs['torch_dtype'])
        self.logger.debug(f'HF using torch_dtype: {torch_dtype}')
        if torch_dtype is not None:
            model_kwargs['torch_dtype'] = torch_dtype

    def _load_model(self,
                    path: str,
                    model_kwargs: dict,
                    peft_path: Optional[str] = None):
        from transformers import AutoModel, AutoModelForCausalLM

        self._set_model_kwargs_torch_dtype(model_kwargs)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                path, **model_kwargs)
        except ValueError:
            self.model = AutoModel.from_pretrained(path, **model_kwargs)

        if peft_path is not None:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model,
                                                   peft_path,
                                                   is_trainable=False)
        self.model.eval()
        self.model.generation_config.do_sample = False

        # A patch for llama when batch_padding = True
        if 'decapoda-research/llama' in path:
            self.model.config.bos_token_id = 1
            self.model.config.eos_token_id = 2
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

    def generate(self,
                 inputs: List[str],
                 max_out_len: int,
                 min_out_len: Optional[int] = None,
                 stopping_criteria: List[str] = [],
                 **kwargs) -> List[str]:
        """Generate results given a list of inputs.

        Args:
            inputs (List[str]): A list of strings.
            max_out_len (int): The maximum length of the output.
            min_out_len (Optional[int]): The minimum length of the output.

        Returns:
            List[str]: A list of generated strings.
        """
        generation_kwargs = kwargs.copy()
        generation_kwargs.update(self.generation_kwargs)
        if self.batch_padding and len(inputs) > 1:
            return self._batch_generate(inputs=inputs,
                                        max_out_len=max_out_len,
                                        min_out_len=min_out_len,
                                        stopping_criteria=stopping_criteria,
                                        **generation_kwargs)
        else:
            return sum(
                (self._single_generate(inputs=[input_],
                                       max_out_len=max_out_len,
                                       min_out_len=min_out_len,
                                       stopping_criteria=stopping_criteria,
                                       **generation_kwargs)
                 for input_ in inputs), [])

    def _batch_generate(self,
                        inputs: List[str],
                        max_out_len: int,
                        min_out_len: Optional[int] = None,
                        stopping_criteria: List[str] = [],
                        **kwargs) -> List[str]:
        """Support for batch prompts inference.

        Args:
            inputs (List[str]): A list of strings.
            max_out_len (int): The maximum length of the output.

        Returns:
            List[str]: A list of generated strings.
        """
        if self.extract_pred_after_decode:
            prompt_lens = [len(input_) for input_ in inputs]

        if self.use_fastchat_template:
            try:
                from fastchat.model import get_conversation_template
            except ModuleNotFoundError:
                raise ModuleNotFoundError(
                    'Fastchat is not implemented. You can use '
                    '\'pip install "fschat[model_worker,webui]"\' '
                    'to implement fastchat.')
            for i in range(len(inputs)):
                conv = get_conversation_template('vicuna')
                conv.append_message(conv.roles[0], inputs[i])
                conv.append_message(conv.roles[1], None)
                inputs[i] = conv.get_prompt()

        # step-1: tokenize the input with batch_encode_plus
        tokens = self.tokenizer.batch_encode_plus(inputs,
                                                  padding=True,
                                                  truncation=True,
                                                  max_length=self.max_seq_len)
        tokens = {
            k: torch.tensor(np.array(tokens[k]), device=self.model.device)
            for k in tokens if k in ['input_ids', 'attention_mask']
        }

        origin_stopping_criteria = stopping_criteria
        if stopping_criteria:
            # Construct huggingface stopping criteria
            if self.tokenizer.eos_token is not None:
                stopping_criteria = stopping_criteria + [
                    self.tokenizer.eos_token
                ]
            stopping_criteria = transformers.StoppingCriteriaList([
                *[
                    MultiTokenEOSCriteria(sequence, self.tokenizer,
                                          tokens['input_ids'].shape[0])
                    for sequence in stopping_criteria
                ],
            ])
            kwargs['stopping_criteria'] = stopping_criteria

        if min_out_len is not None:
            kwargs['min_new_tokens'] = min_out_len

        # step-2: conduct model forward to generate output
        outputs = self.model.generate(**tokens,
                                      max_new_tokens=max_out_len,
                                      **kwargs)

        if not self.extract_pred_after_decode:
            outputs = outputs[:, tokens['input_ids'].shape[1]:]

        decodeds = self.tokenizer.batch_decode(outputs,
                                               skip_special_tokens=True)

        if self.extract_pred_after_decode:
            decodeds = [
                token[len_:] for token, len_ in zip(decodeds, prompt_lens)
            ]

        if self.end_str:
            decodeds = [token.split(self.end_str)[0] for token in decodeds]
        if origin_stopping_criteria:
            for t in origin_stopping_criteria:
                decodeds = [token.split(t)[0] for token in decodeds]
        return decodeds

    def _single_generate(self,
                         inputs: List[str],
                         max_out_len: int,
                         min_out_len: Optional[int] = None,
                         stopping_criteria: List[str] = [],
                         **kwargs) -> List[str]:
        """Support for single prompt inference.

        Args:
            inputs (List[str]): A list of strings.
            max_out_len (int): The maximum length of the output.

        Returns:
            List[str]: A list of generated strings.
        """
        if self.extract_pred_after_decode:
            prompt_lens = [len(input_) for input_ in inputs]

        if self.use_fastchat_template:
            try:
                from fastchat.model import get_conversation_template
            except ModuleNotFoundError:
                raise ModuleNotFoundError(
                    'Fastchat is not implemented. You can use '
                    '\'pip install "fschat[model_worker,webui]"\' '
                    'to implement fastchat.')
            conv = get_conversation_template('vicuna')
            conv.append_message(conv.roles[0], inputs[0])
            conv.append_message(conv.roles[1], None)
            inputs = [conv.get_prompt()]

        if self.mode == 'mid':
            input_ids = self.tokenizer(inputs, truncation=False)['input_ids']
            input_ids = torch.tensor(input_ids, device=self.model.device)
            if len(input_ids[0]) > self.max_seq_len - max_out_len:
                half = int((self.max_seq_len - max_out_len) / 2)
                inputs = [
                    self.tokenizer.decode(input_ids[0][:half],
                                          skip_special_tokens=True) +
                    self.tokenizer.decode(input_ids[0][-half:],
                                          skip_special_tokens=True)
                ]

        input_ids = self.tokenizer(inputs,
                                   truncation=True,
                                   max_length=self.max_seq_len -
                                   max_out_len)['input_ids']
        input_ids = torch.tensor(input_ids, device=self.model.device)
        origin_stopping_criteria = stopping_criteria
        if stopping_criteria:
            # Construct huggingface stopping criteria
            if self.tokenizer.eos_token is not None:
                stopping_criteria = stopping_criteria + [
                    self.tokenizer.eos_token
                ]
            stopping_criteria = transformers.StoppingCriteriaList([
                *[
                    MultiTokenEOSCriteria(sequence, self.tokenizer,
                                          input_ids.shape[0])
                    for sequence in stopping_criteria
                ],
            ])
            kwargs['stopping_criteria'] = stopping_criteria

        if min_out_len is not None:
            kwargs['min_new_tokens'] = min_out_len

        # To accommodate the PeftModel, parameters should be passed in
        # key-value format for generate.
        outputs = self.model.generate(input_ids=input_ids,
                                      max_new_tokens=max_out_len,
                                      **kwargs)

        if not self.extract_pred_after_decode:
            outputs = outputs[:, input_ids.shape[1]:]

        decodeds = self.tokenizer.batch_decode(outputs,
                                               skip_special_tokens=True)

        if self.extract_pred_after_decode:
            decodeds = [
                token[len_:] for token, len_ in zip(decodeds, prompt_lens)
            ]

        if self.end_str:
            decodeds = [token.split(self.end_str)[0] for token in decodeds]
        if origin_stopping_criteria:
            for t in origin_stopping_criteria:
                decodeds = [token.split(t)[0] for token in decodeds]
        return decodeds

    def get_logits(self, inputs: List[str]):
        # print(f"inputs: {inputs}")
        if self.batch_padding and len(inputs) > 1:
            # batch inference
            tokens = self.tokenizer(inputs,
                                    padding=True,
                                    truncation=True,
                                    max_length=self.max_seq_len)

            tokens = {
                k: torch.tensor(np.array(tokens[k]), device=self.model.device)
                for k in tokens if k in ['input_ids', 'attention_mask']
            }
            outputs = self.model(**tokens)

        else:
            input_ids = self.tokenizer(
                inputs,
                padding=False,
                truncation=True,
                max_length=self.max_seq_len)['input_ids']
            input_ids = torch.tensor(input_ids, device=self.model.device)
            tokens = {'input_ids': input_ids}
            outputs = self.model(input_ids)
            # print(f"outputs[0]:{outputs[0].shape}")
            # print(f"input_ids : {input_ids.shape}")
        return outputs[0], {'tokens': tokens}

    def get_ppl(self,
                inputs: List[str],
                mask_length: Optional[List[int]] = None) -> List[float]:
        """Get perplexity scores given a list of inputs.

        Args:
            inputs (List[str]): A list of strings.
            mask_length (Optional[List[int]]): A list of mask lengths. If
                provided, the perplexity scores will be calculated with the
                first mask_length[i] tokens masked out. It's okay to skip
                its implementation if advanced features in PPLInfernecer is
                not needed.

        Returns:
            List[float]: A list of perplexity scores.
        """

        if self.batch_padding and len(inputs) > 1:
            assert self.tokenizer.pad_token
            return self._get_ppl(inputs, mask_length=mask_length)
        else:
            return np.concatenate([
                self._get_ppl(inputs=[text], mask_length=mask_length)
                for text in inputs
            ])

    def _get_ppl(self,
                 inputs: List[str],
                 mask_length: Optional[List[int]] = None) -> List[float]:
        """Get perplexity scores given a list of inputs.

        Args:
            inputs (List[str]): A list of strings.
            mask_length (Optional[List[int]]): A list of mask lengths. If
                provided, the perplexity scores will be calculated with the
                first mask_length[i] tokens masked out. It's okay to skip
                its implementation if advanced features in PPLInfernecer is
                not needed.

        Returns:
            List[float]: A list of perplexity scores.
        """

        outputs, inputs = self.get_logits(inputs)
        shift_logits = outputs[..., :-1, :].contiguous().float()
        # print(f"shift_logits: {shift_logits.shape}")
        shift_labels = inputs['tokens']['input_ids'][..., 1:].contiguous()
        # print(f"shift_labels: {shift_labels.shape}")
        loss_fct = torch.nn.CrossEntropyLoss(
            reduction='none', ignore_index=self.tokenizer.pad_token_id)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)).view(shift_labels.size())

        if mask_length is not None:
            mask = torch.zeros_like(shift_labels)  # [batch,seqlen]
            for i in range(len(mask)):
                for j in range(mask_length[i] - 1, len(mask[i])):
                    mask[i][j] = 1
            loss = loss * mask

        lens = (inputs['tokens']['input_ids'] !=
                self.tokenizer.pad_token_id).sum(-1).cpu().numpy()
        if mask_length is not None:
            lens -= np.array(mask_length)
        ce_loss = loss.float().sum(-1).cpu().detach().numpy() / lens
        return ce_loss

    def get_loglikelihood(
            self,
            inputs: List[str],
            conts: List[str],
            mask_length: Optional[List[int]] = None) -> List[float]:
        """Get loglikelihood scores given a list of inputs.

        Args:
            inputs (List[str]): A list of strings.
            conts (List[str]): A list of strings: slices after the space.
            NOT SUPPORT mask_length YET!
            mask_length (Optional[List[int]]): A list of mask lengths. If
                provided, the perplexity scores will be calculated with the
                first mask_length[i] tokens masked out. It's okay to skip
                its implementation if advanced features in PPLInfernecer is
                not needed.

        Returns:
            List[float]: A list of loglikelihood scores.
        """
        assert mask_length is None, 'Not support mask_length yet.'
        if self.batch_padding and len(inputs) > 1:
            assert self.tokenizer.pad_token
            return self._get_loglikelihood(inputs, conts)
        else:
            return np.concatenate([
                self._get_loglikelihood(inputs=[inputs[idx]],
                                        conts=[conts[idx]])
                for idx in range(len(inputs))
            ])

    def _get_loglikelihood(self, inputs: str, conts: str) -> float:
        """Get loglikelihood scores given input string and continuation string.

        Args:
            inputs (str): string.
            conts (str): strings: slices after the space.
        Returns:
            float: loglikelihood scores.
        """
        input_tokenizer_out = self.tokenizer(inputs,
                                             padding=True,
                                             truncation=False,
                                             return_length=True,
                                             return_tensors='pt').to(
                                                 self.model.device)

        input_ids = input_tokenizer_out['input_ids'][:, :self.max_seq_len]
        input_length = input_tokenizer_out['length']
        context_ids = [
            self.tokenizer(inputs[i].replace(conts[i], ''),
                           padding=False,
                           truncation=True,
                           max_length=self.max_seq_len)['input_ids']
            for i in range(len(inputs))
        ]
        # forward
        outputs = self.model(input_ids)['logits']
        outputs = torch.nn.functional.log_softmax(outputs, dim=-1)
        # calculate loglikelihood
        answer = np.zeros(len(inputs))
        for i in range(len(inputs)):
            if self.tokenizer.padding_side == 'right':
                cont_ids = input_ids[i, len(context_ids[i]):input_length[i]]
                logits = outputs[i,
                                 len(context_ids[i]) - 1:input_length[i] -
                                 1, :]  # noqa
            else:
                cont_ids = input_ids[i, len(context_ids[i]) - input_length[i]:]
                logits = outputs[i,
                                 len(context_ids[i]) - input_length[i] - 1:-1]
            # Reducing the dimension will lead to a wrong outcome
            logits_gather = torch.gather(
                logits.unsqueeze(0), 2,
                cont_ids.unsqueeze(0).unsqueeze(-1))  # [1, seq]
            # Answer: sum the likelihood of each token in continuation
            answer[i] = float(logits_gather.detach().cpu().sum())
        return answer

    def get_mink_percent(self, inputs: List[str], k: int = 20) -> List[float]:
        """https://swj0419.github.io/detect-pretrain.github.io/"""

        if self.batch_padding and len(inputs) > 1:
            assert self.tokenizer.pad_token
            return self._get_mink_percent(inputs, k=k)
        else:
            return np.concatenate([
                self._get_mink_percent(inputs=[text], k=k) for text in inputs
            ])

    def _get_mink_percent(self, inputs: List[str], k: int = 20) -> List[float]:
        outputs, inputs = self.get_logits(inputs)
        shift_logits = outputs[:, :-1, :].contiguous().float()
        shift_labels = inputs['tokens']['input_ids'][:, 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(
            reduction='none', ignore_index=self.tokenizer.pad_token_id)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)).view(shift_labels.size())
        lens = (inputs['tokens']['input_ids'] !=
                self.tokenizer.pad_token_id).sum(-1).cpu().numpy()
        mink_percent = []
        for nloss, nlen in zip(loss, lens):
            nlen = int(nlen)
            minklen = max(nlen * k // 100, 1)
            nloss = torch.topk(loss[-nlen:], minklen, dim=-1)[0]
            nloss = -nloss.float().mean().cpu().detach().numpy()
            mink_percent.append(nloss)
        return np.array(mink_percent)

    def get_token_len(self, prompt: str) -> int:
        """Get lengths of the tokenized strings.

        Args:
            prompt (str): Input string.

        Returns:
            int: Length of the input tokens
        """
        return len(self.tokenizer.encode(prompt))


@MODELS.register_module()
class HuggingFaceCausalLM(HuggingFace):
    """Model wrapper around HuggingFace CausalLM.

    Args:
        path (str): The name or path to HuggingFace's model.
        hf_cache_dir: Set the cache dir to HF model cache dir. If None, it will
            use the env variable HF_MODEL_HUB. Defaults to None.
        max_seq_len (int): The maximum length of the input sequence. Defaults
            to 2048.
        tokenizer_path (str): The path to the tokenizer. Defaults to None.
        tokenizer_kwargs (dict): Keyword arguments for the tokenizer.
            Defaults to {}.
        peft_path (str, optional): The name or path to the HuggingFace's PEFT
            model. If None, the original model will not be converted to PEFT.
            Defaults to None.
        tokenizer_only (bool): If True, only the tokenizer will be initialized.
            Defaults to False.
        model_kwargs (dict): Keyword arguments for the model, used in loader.
            Defaults to dict(device_map='auto').
        meta_template (Dict, optional): The model's meta prompt
            template if needed, in case the requirement of injecting or
            wrapping of any meta instructions.
        batch_padding (bool): If False, inference with be performed in for-loop
            without batch padding.
    """

    def _load_model(self,
                    path: str,
                    model_kwargs: dict,
                    peft_path: Optional[str] = None):
        from transformers import AutoModelForCausalLM

        self._set_model_kwargs_torch_dtype(model_kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)
        if peft_path is not None:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model,
                                                   peft_path,
                                                   is_trainable=False)
        self.model.eval()
        self.model.generation_config.do_sample = False


class HuggingFaceChatGLM3(HuggingFace):
    """Model wrapper around HuggingFace's ChatGLM3. Details available in
    `https://huggingface.co/THUDM/chatglm3-6b`.

    model.chat() is used for inference.
    """

    def __init__(self,
                 path: str,
                 hf_cache_dir: Optional[str] = None,
                 max_seq_len: int = 2048,
                 tokenizer_path: Optional[str] = None,
                 tokenizer_kwargs: dict = dict(),
                 peft_path: Optional[str] = None,
                 tokenizer_only: bool = False,
                 model_kwargs: dict = dict(device_map='auto'),
                 generation_kwargs: dict = dict(),
                 meta_template: Optional[Dict] = None,
                 extract_pred_after_decode: bool = False,
                 batch_padding: bool = False,
                 pad_token_id: Optional[int] = None,
                 mode: str = 'none',
                 num_extra_tokens: int = 50):
        super().__init__(path=path,
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
                         mode=mode)
        self.template_parser = APITemplateParser(meta_template)
        # used to compensate for #tokens occupied by sth like system prompt
        self.num_extra_tokens = num_extra_tokens

    def generate(self,
                 inputs: List[PromptType],
                 max_out_len: int = 512,
                 skip_overlength=False,
                 **kwargs) -> str:
        """Generate response from input prompt.

        Args:
            inputs (list): input prompt
            max_out_len (int): max output length
        """
        generation_kwargs = kwargs.copy()
        generation_kwargs.update(self.generation_kwargs)

        responses = []
        for _input in inputs:
            assert isinstance(_input, (str, PromptList))
            if isinstance(_input, str):
                history = [{'role': 'user', 'content': _input}]
            else:
                history = []
                for item in _input:
                    msg = {
                        'content': item['prompt'],
                        'role': {
                            'HUMAN': 'user',
                            'BOT': 'assistant',
                            'SYSTEM': 'system',
                        }[item['role'].upper()]
                    }
                    history.append(msg)
            user_content = history[-1]['content']
            history = history[:-1]

            if skip_overlength:
                # The model will report the following error
                # if the sequence length is greater than the maximum length:
                # "Input length of input_ids is {INPUT_IDS},
                # but `max_length` is set to 8192.
                # This can lead to unexpected behavior.
                # You should consider increasing `max_new_tokens`."
                # The following hardcode can fix this exception.
                len_user_content = len(self.tokenizer.encode(user_content))
                if len_user_content > 8192:
                    responses.append('')
                    continue

            response, history = self.model.chat(self.tokenizer,
                                                user_content,
                                                history=history,
                                                max_new_tokens=max_out_len,
                                                **generation_kwargs)
            # response will be dict sometime
            if isinstance(response, dict):
                response = response.get('content', '')
            responses.append(response)
        return responses

    def get_token_len(self, prompt: str) -> int:
        return len(self.tokenizer.encode(prompt)) + self.num_extra_tokens


_DYNAMIC_MOE_DEFAULT_TSV_PATH = 'dynamic_moe_routing_token_stats_piqa.tsv'


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
        moe_package_name: str = "Predict_MoE",
        moe_modeling_module: str = "modeling_moe_ori",
        moe_config_module: str = "configuration_moe",
    ):
        # ⚠️ 关键：在调用父类 __init__ 之前提取并移除自定义参数
        # 避免这些参数被传递到 HuggingFace 的模型加载函数
        enable_expert_stats = model_kwargs.pop('enable_expert_stats', False)
        expert_stats_path = model_kwargs.pop('expert_stats_path', './expert_usage_stats.json')
        self.moe_package_name = moe_package_name
        self.moe_modeling_module = moe_modeling_module
        self.moe_config_module = moe_config_module
        
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

        import os
        import sys

        from transformers import LlamaTokenizer

        # 将模型目录和其子目录加入 sys.path，避免动态 MoE 包因路径差异导入失败
        moe_root = os.path.abspath(path)
        print(f"----------path: {path} -------------------")
        moe_inner = os.path.join(moe_root, self.moe_package_name)
        for candidate in (moe_root, moe_inner):
            if os.path.isdir(candidate) and candidate not in sys.path:
                sys.path.append(candidate)

        import importlib
        modeling_candidates = [
            f"{self.moe_package_name}.modeling.{self.moe_modeling_module}",
        ]
        config_candidates = [
            f"{self.moe_package_name}.modeling.{self.moe_config_module}",
        ]

        # 兼容模型目录中同时包含包目录和直接 modeling/ 目录两种结构
        if os.path.isdir(os.path.join(moe_root, "modeling")):
            modeling_candidates.append(f"modeling.{self.moe_modeling_module}")
            config_candidates.append(f"modeling.{self.moe_config_module}")

        moe_module = None
        cfg_module = None
        import_errors = []
        try:
            for modeling_name in modeling_candidates:
                try:
                    moe_module = importlib.import_module(modeling_name)
                    break
                except ImportError as e:
                    import_errors.append((modeling_name, str(e)))
            for config_name in config_candidates:
                try:
                    cfg_module = importlib.import_module(config_name)
                    break
                except ImportError as e:
                    import_errors.append((config_name, str(e)))
            if moe_module is None or cfg_module is None:
                raise ImportError("failed to import requested Dynamic MoE modules")
        except ImportError as e:
            raise ImportError(
                "Cannot import Dynamic_MoE modules. "
                f"Tried modeling={modeling_candidates}, config={config_candidates}. "
                f"Collected import errors: {import_errors}. "
                f"Original error: {e}"
            )
        print(
            "modeling moe module: "
            f"{moe_module.__name__}, config module: {cfg_module.__name__}, path = {path}"
        )
        MoEForCausalLM = getattr(moe_module, "MoEForCausalLM")
        MoEConfig = getattr(cfg_module, "MoEConfig")
        
        self.tokenizer = LlamaTokenizer.from_pretrained(path)
        self.tokenizer.pad_token = self.tokenizer.unk_token

        model_config = MoEConfig.from_pretrained(path, trust_remote_code=True)
        # 将统计功能配置传递给 MoE config
        if enable_expert_stats:
            setattr(model_config, "enable_expert_stats", True)
        
        self.model = MoEForCausalLM.from_pretrained(
            path,
            from_tf=False,
            config=model_config,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).cuda()
        self.model.eval()
        
        # 检测是否为 modeling_moe_infer（Predict MoE），启用统计功能
        self.is_predict_moe = "modeling_moe_infer" in moe_module.__name__
        if self.is_predict_moe and enable_expert_stats:
            print("[INFO] Detected Predict MoE model, enabling expert usage statistics")
            self._stats_output_path = expert_stats_path

    def get_ppl(self, inputs: List[str], mask_length=None):
        """Compute PPL scores while recording per-token MoE routing metrics.

        Hooks into every SwitchMLP.router (a Linear submodule) whose
        ``use_switch`` flag is True to capture routing probabilities without
        storing the full (seq_len, batch, num_experts) tensor.  Only 3 scalar
        summary stats per token per layer are kept.

        The TSV path defaults to _DYNAMIC_MOE_DEFAULT_TSV_PATH and can be
        overridden by setting ``self._routing_tsv_path`` before calling.
        """
        assert self.tokenizer.pad_token, 'pad_token must be set'
        pad_token_id = self.tokenizer.pad_token_id

        tokens = self.tokenizer.batch_encode_plus(
            inputs,
            return_tensors='pt',
            padding=True,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_seq_len,
        )
        tokens = {k: v.to(self.model.device) for k, v in tokens.items()}

        # ------------------------------------------------------------------
        # Register forward hooks on SwitchMLP.router (Linear submodule).
        # hidden_states fed to router: (seq_len, batch, hidden)
        # router output (our hook's `out`): (seq_len, batch, num_experts) raw logits
        # We apply softmax at dim=2, permute to (batch, seq_len, num_experts),
        # flatten to (batch*seq_len, num_experts) for consistent stat_idx.
        # ------------------------------------------------------------------
        routing_stats = {}   # lkey -> list[(top4, delta45, entropy)]
        layer_indices = {}   # lkey -> int
        hooks = []

        for _name, module in self.model.named_modules():
            if (type(module).__name__ == 'SwitchMLP'
                    and getattr(module, 'use_switch', False)):
                layer_idx = module.layer_num
                layer_key = f'layer{layer_idx}'
                layer_indices[layer_key] = layer_idx

                def make_hook(lkey):
                    def hook_fn(mod, inp, out):
                        # out: (seq_len, batch, num_experts) raw logits
                        rw = torch.nn.functional.softmax(out.float(), dim=2)
                        rw = rw.permute(1, 0, 2)  # (batch, seq_len, num_experts)
                        rw_np = rw.reshape(-1, rw.size(2)).cpu().detach().numpy()
                        stats = []
                        for token_probs in rw_np:
                            sp = sorted(token_probs, reverse=True)
                            s_top4 = float(sp[0] + sp[1] + sp[2] + sp[3]) \
                                if len(sp) >= 4 else float(sum(sp))
                            delta = float(sp[3] - sp[4]) if len(sp) >= 5 else 0.0
                            ent = 0.0
                            for p in token_probs:
                                p = float(p)
                                if p > 1e-12:
                                    ent -= p * math.log(p)
                            stats.append((s_top4, delta, ent))
                        routing_stats[lkey] = stats
                    return hook_fn

                hooks.append(
                    module.router.register_forward_hook(make_hook(layer_key)))

        # ------------------------------------------------------------------
        # Forward pass - hooks fire here.
        # ------------------------------------------------------------------
        try:
            with torch.no_grad():
                outputs = self.model(
                    input_ids=tokens['input_ids'],
                    attention_mask=tokens.get('attention_mask'),
                    return_dict=False,
                )
                logits = outputs[0]  # (batch, seq_len, vocab_size)
        finally:
            for h in hooks:
                h.remove()

        # ------------------------------------------------------------------
        # Compute per-token cross-entropy losses.
        # ------------------------------------------------------------------
        batch_size, seq_len, vocab_size = logits.shape
        shift_logits = logits[:, :-1, :].contiguous().float()
        shift_labels = tokens['input_ids'][:, 1:].contiguous()
        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1),
            ignore_index=pad_token_id,
            reduction='none',
        ).view(batch_size, seq_len - 1)   # (batch, seq_len-1)
        lens = (tokens['input_ids'] != pad_token_id).sum(-1).cpu().numpy()

        if mask_length is not None:
            mask = torch.zeros_like(shift_labels)
            for i in range(len(mask)):
                for j in range(mask_length[i] - 1, len(mask[i])):
                    mask[i][j] = 1
            loss = loss * mask
            lens -= np.array(mask_length)

        ce_loss = loss.float().sum(-1).cpu().detach().numpy() / lens

        # ------------------------------------------------------------------
        # Write per-token routing stats to TSV.
        # ------------------------------------------------------------------
        self._write_token_stats_tsv(
            inputs=inputs,
            token_ids_np=tokens['input_ids'].cpu().numpy(),
            loss_np=loss.float().cpu().detach().numpy(),
            routing_stats=routing_stats,
            layer_indices=layer_indices,
            batch_size=batch_size,
            seq_len=seq_len,
            pad_token_id=pad_token_id,
        )

        return ce_loss

    def _write_token_stats_tsv(
        self,
        inputs,
        token_ids_np,
        loss_np,
        routing_stats,
        layer_indices,
        batch_size,
        seq_len,
        pad_token_id,
    ):
        """Append per-token routing stats rows to the TSV file."""
        sorted_layers = sorted(layer_indices.items(), key=lambda x: x[1])
        all_lkeys = [lk for lk, _ in sorted_layers]
        last_2_lkeys = (
            {lk for lk, _ in sorted_layers[-2:]}
            if len(sorted_layers) >= 2
            else set(all_lkeys)
        )

        tsv_path = getattr(self, '_routing_tsv_path', _DYNAMIC_MOE_DEFAULT_TSV_PATH)
        write_header = not os.path.exists(tsv_path)

        with open(tsv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter='\t')
            if write_header:
                per_layer_header = []
                for lkey in all_lkeys:
                    per_layer_header += [
                        f'top4_{lkey}', f'delta_{lkey}', f'entropy_{lkey}',
                    ]
                writer.writerow([
                    'input_text', 'token_idx', 'token_nll',
                    'top4_all', 'delta_all', 'entropy_all',
                    'top4_last2', 'delta_last2', 'entropy_last2',
                ] + per_layer_header)

            for i in range(batch_size):
                num_valid = int((token_ids_np[i] != pad_token_id).sum())
                label = inputs[i][:60].replace('\t', ' ').replace('\n', ' ')

                for t in range(num_valid - 1):
                    nll = float(loss_np[i, t])
                    if token_ids_np[i, t + 1] == pad_token_id:
                        continue

                    stat_idx = i * seq_len + t

                    top4_all_v, delta_all_v, ent_all_v = [], [], []
                    top4_l2_v, delta_l2_v, ent_l2_v = [], [], []
                    per_layer_vals = []

                    for lkey in all_lkeys:
                        if lkey not in routing_stats:
                            per_layer_vals += ['', '', '']
                            continue
                        stats = routing_stats[lkey]
                        if stat_idx >= len(stats):
                            per_layer_vals += ['', '', '']
                            continue
                        s_top4, delta, ent = stats[stat_idx]
                        top4_all_v.append(s_top4)
                        delta_all_v.append(delta)
                        ent_all_v.append(ent)
                        if lkey in last_2_lkeys:
                            top4_l2_v.append(s_top4)
                            delta_l2_v.append(delta)
                            ent_l2_v.append(ent)
                        per_layer_vals += [
                            f'{s_top4:.6f}', f'{delta:.6f}', f'{ent:.6f}',
                        ]

                    if not top4_all_v:
                        continue

                    def _mean(lst):
                        return sum(lst) / len(lst)

                    writer.writerow([
                        label,
                        t,
                        f'{nll:.6f}',
                        f'{_mean(top4_all_v):.6f}',
                        f'{_mean(delta_all_v):.6f}',
                        f'{_mean(ent_all_v):.6f}',
                        f'{_mean(top4_l2_v):.6f}' if top4_l2_v else '',
                        f'{_mean(delta_l2_v):.6f}' if delta_l2_v else '',
                        f'{_mean(ent_l2_v):.6f}' if ent_l2_v else '',
                    ] + per_layer_vals)

    def generate(
        self,
        inputs: List[PromptType],
        max_out_len: int = 512,
        **kwargs,
    ) -> str:
        tokens = self.tokenizer(
            inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_seq_len,
        )
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
            # do_sample=True,
            # dynamic_k=[2,0,6,7,3,6,4,7,7,4,1,3,6,2,7,5,5,4,1,4,2,5,1,4,2,7,7,7,1,0,5,7],
        )
        outputs = self.tokenizer.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        response = [outputs[i][len(inputs[i]) :] for i in range(len(outputs))][0]
        return response
    
    def get_expert_usage_stats(self):
        """收集所有层的专家使用统计（仅在 Predict MoE 中生效）"""
        if not getattr(self, 'is_predict_moe', False):
            return None
        
        if not hasattr(self.model, 'model') or not hasattr(self.model.model, 'layers'):
            return None
        
        stats = {
            'model_type': 'predict_moe',
            'per_layer': [],
            'global_avg_k': 0.0,
            'total_tokens': 0
        }
        
        total_k_sum = 0
        for idx, layer in enumerate(self.model.model.layers):
            if hasattr(layer.mlp, 'get_expert_usage_stats'):
                layer_stats = layer.mlp.get_expert_usage_stats()
                if layer_stats and layer_stats['total_tokens'] > 0:
                    stats['per_layer'].append({
                        'layer_idx': idx,
                        **layer_stats
                    })
                    total_k_sum += layer_stats['avg_k'] * layer_stats['total_tokens']
                    stats['total_tokens'] += layer_stats['total_tokens']
        
        if stats['total_tokens'] > 0:
            stats['global_avg_k'] = total_k_sum / stats['total_tokens']
        
        return stats
    
    def save_expert_usage_stats(self, output_path=None):
        """保存专家使用统计到文件"""
        if not getattr(self, 'is_predict_moe', False):
            print("[WARN] Not a Predict MoE model, skipping stats saving")
            return
        
        stats = self.get_expert_usage_stats()
        if stats is None:
            print("[WARN] No expert usage statistics collected (stats is None)")
            return
        
        if stats.get('total_tokens', 0) == 0:
            print("[WARN] No tokens processed yet, cannot save statistics")
            return
        
        if output_path is None:
            output_path = getattr(self, '_stats_output_path', './expert_usage_stats.json')
        
        import json
        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"[INFO] Expert usage stats saved to {output_path}")
        print(f"[INFO] Global average experts per token: {stats['global_avg_k']:.2f}")
    
    def reset_expert_usage_stats(self):
        """重置所有层的统计信息"""
        if not getattr(self, 'is_predict_moe', False):
            return
        
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            for layer in self.model.model.layers:
                if hasattr(layer.mlp, 'reset_expert_usage_stats'):
                    layer.mlp.reset_expert_usage_stats()


class HuggingFacePredictMoE(HuggingFace):

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
        # ⚠️ 关键：在调用父类 __init__ 之前提取并移除自定义参数
        # 避免这些参数被传递到 HuggingFace 的模型加载函数
        enable_expert_stats = model_kwargs.pop('enable_expert_stats', False)
        expert_stats_path = model_kwargs.pop('expert_stats_path', './expert_usage_stats.json')
        
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

        import os
        import sys

        from transformers import LlamaTokenizer

        # 将模型目录和其子目录加入 sys.path，避免动态 MoE 包因路径差异导入失败
        moe_root = os.path.abspath(path)
        print(f"----------path: {path} -------------------")
        moe_inner = os.path.join(moe_root, "Predict_MoE")
        for candidate in (moe_root, moe_inner):
            if os.path.isdir(candidate) and candidate not in sys.path:
                sys.path.append(candidate)

        import importlib
        # 兼容两种包结构的导入路径
        try:
            # e.g. {path}/Dynamic_MoE/modeling/...
            moe_module = importlib.import_module(
                "Predict_MoE.modeling.modeling_moe_infer"
            )
            cfg_module = importlib.import_module(
                "Predict_MoE.modeling.configuration_moe"
            )
        except ImportError as e:
            # 这里直接报错，方便调试，而不是再去导入一个不存在的路径
            raise ImportError(
                "Cannot import Dynamic_MoE modules. "
                "Please make sure Dynamic_MoE is installed in the current environment "
                "and can be imported as 'import Dynamic_MoE'. "
                f"Original error: {e}"
            )
        print(f"modeling moe is : modeling moe infer, path = {path}")
        MoEForCausalLM = getattr(moe_module, "MoEForCausalLM")
        MoEConfig = getattr(cfg_module, "MoEConfig")
        
        self.tokenizer = LlamaTokenizer.from_pretrained(path)
        self.tokenizer.pad_token = self.tokenizer.unk_token

        model_config = MoEConfig.from_pretrained(path, trust_remote_code=True)
        # 将统计功能配置传递给 MoE config
        if enable_expert_stats:
            setattr(model_config, "enable_expert_stats", True)
        
        self.model = MoEForCausalLM.from_pretrained(
            path,
            from_tf=False,
            config=model_config,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).cuda()
        self.model.eval()
        
        # 检测是否为 modeling_moe_infer（Predict MoE），启用统计功能
        self.is_predict_moe = "modeling_moe_infer" in moe_module.__name__
        if self.is_predict_moe and enable_expert_stats:
            print("[INFO] Detected Predict MoE model, enabling expert usage statistics")
            self._stats_output_path = expert_stats_path

    def generate(
        self,
        inputs: List[PromptType],
        max_out_len: int = 512,
        **kwargs,
    ) -> str:
        tokens = self.tokenizer(
            inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_seq_len,
        )
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
            # do_sample=True,
            # dynamic_k=[2,0,6,7,3,6,4,7,7,4,1,3,6,2,7,5,5,4,1,4,2,5,1,4,2,7,7,7,1,0,5,7],
        )
        outputs = self.tokenizer.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        response = [outputs[i][len(inputs[i]) :] for i in range(len(outputs))][0]
        return response
    
    def get_expert_usage_stats(self):
        """收集所有层的专家使用统计（仅在 Predict MoE 中生效）"""
        if not getattr(self, 'is_predict_moe', False):
            return None
        
        if not hasattr(self.model, 'model') or not hasattr(self.model.model, 'layers'):
            return None
        
        stats = {
            'model_type': 'predict_moe',
            'per_layer': [],
            'global_avg_k': 0.0,
            'total_tokens': 0
        }
        
        total_k_sum = 0
        for idx, layer in enumerate(self.model.model.layers):
            if hasattr(layer.mlp, 'get_expert_usage_stats'):
                layer_stats = layer.mlp.get_expert_usage_stats()
                if layer_stats and layer_stats['total_tokens'] > 0:
                    stats['per_layer'].append({
                        'layer_idx': idx,
                        **layer_stats
                    })
                    total_k_sum += layer_stats['avg_k'] * layer_stats['total_tokens']
                    stats['total_tokens'] += layer_stats['total_tokens']
        
        if stats['total_tokens'] > 0:
            stats['global_avg_k'] = total_k_sum / stats['total_tokens']
        
        return stats
    
    def save_expert_usage_stats(self, output_path=None):
        """保存专家使用统计到文件"""
        if not getattr(self, 'is_predict_moe', False):
            print("[WARN] Not a Predict MoE model, skipping stats saving")
            return
        
        stats = self.get_expert_usage_stats()
        if stats is None:
            print("[WARN] No expert usage statistics collected (stats is None)")
            return
        
        if stats.get('total_tokens', 0) == 0:
            print("[WARN] No tokens processed yet, cannot save statistics")
            return
        
        if output_path is None:
            output_path = getattr(self, '_stats_output_path', './expert_usage_stats.json')
        
        import json
        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"[INFO] Expert usage stats saved to {output_path}")
        print(f"[INFO] Global average experts per token: {stats['global_avg_k']:.2f}")
    
    def reset_expert_usage_stats(self):
        """重置所有层的统计信息"""
        if not getattr(self, 'is_predict_moe', False):
            return
        
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            for layer in self.model.model.layers:
                if hasattr(layer.mlp, 'reset_expert_usage_stats'):
                    layer.mlp.reset_expert_usage_stats()
