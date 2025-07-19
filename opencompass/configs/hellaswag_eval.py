from opencompass.utils.text_postprocessors import first_capital_postprocess
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import HellaswagDataset
from opencompass.openicl.icl_evaluator import AccwithDetailsEvaluator
datasets = [
    dict(
        type=HellaswagDataset,
        path='/mnt/data/hellaswag',
        reader_cfg=dict(
            input_columns=['ctx', 'endings'],
            output_column='label',
        ),
        infer_cfg=dict(
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(
                    round=[
                        dict(
                            role='HUMAN',
                            prompt="Choose the correct ending for the following sentence:\n{ctx}\nOptions:\nA. {ending0}\nB. {ending1}\nC. {ending2}\nD. {ending3}\nAnswer:"
                        )
                    ]
                ),
            ),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(type=GenInferencer),
        ),
        eval_cfg=dict(
            evaluator=dict(type=AccwithDetailsEvaluator),
            pred_role='BOT',
            pred_postprocessor=dict(type=first_capital_postprocess)
        )
    )
]

models = [
    dict(
        type='opencompass.models.HuggingFaceCausalLM',
        abbr='Dynamic-MoE',
        path='/mnt/data/Dynamic_moe',
        tokenizer_path='/mnt/data/Dynamic_moe',
        model_kwargs=dict(
            device_map='auto',
            trust_remote_code=True,
            torch_dtype='auto'
        ),
        max_out_len=512,
        batch_size=8,
        run_cfg=dict(num_gpus=1),
    )
]