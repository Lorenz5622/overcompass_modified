# # configs/piqa_eval.py

# from opencompass.utils.text_postprocessors import first_capital_postprocess
# from opencompass.openicl.icl_prompt_template import PromptTemplate
# from opencompass.openicl.icl_retriever import ZeroRetriever
# from opencompass.openicl.icl_inferencer import GenInferencer
# from opencompass.openicl.icl_evaluator import AccEvaluator
# from opencompass.datasets import PIQADatasetV2
# from opencompass.utils.text_postprocessors import first_option_postprocess
# piqa_reader_cfg = dict(
#     input_columns=['goal', 'sol1', 'sol2'],
#     output_column='answer',
#     test_split='validation')

# datasets = [
#     dict(
#         type='opencompass.datasets.PIQADataset',
#         path='/mnt/data/piqa',
#         reader_cfg=dict(
#             input_columns=['goal', 'sol1', 'sol2'],
#             output_column='answer',
#             test_split='validation'
#         ),
#         infer_cfg=dict(
#             prompt_template=dict(
#                 type=PromptTemplate,
#                 template=dict(
#                     round=[
#                         dict(
#                             role='HUMAN',
#                             prompt='{goal}\nA. {sol1}\nB. {sol2}\nAnswer:')
#                     ], ),
#             ),
#             retriever=dict(type=ZeroRetriever),
#             inferencer=dict(type=GenInferencer),
#         ),
#         eval_cfg = dict(
#             evaluator=dict(type=AccEvaluator),
#             pred_role='BOT',
#             pred_postprocessor=dict(type=first_option_postprocess, options='AB'),
#         ),
#     )
# ]

# models = [
#     dict(
#         type='opencompass.models.HuggingFaceCausalLM',
#         abbr='Dynamic-MoE',
#         path='/mnt/data/models/Dynamic_moe',
#         tokenizer_path='/mnt/data/models/Dynamic_moe',
#         model_kwargs=dict(
#             device_map='auto',
#             trust_remote_code=True,
#             torch_dtype='auto'
#         ),
#         max_out_len=100,
#         batch_size=8,
#         run_cfg=dict(num_gpus=1),
#     )
# ]

from opencompass.models import HuggingFace

models = [
    dict(
        type=HuggingFace,
        abbr='Dynamic_moe',
        path='/mnt/data/Dynamic_moe',
        tokenizer_path='/mnt/data/Dynamic_moe',
        model_kwargs=dict(device_map="auto"),
        max_out_len=512,
        batch_size=8,
        run_cfg=dict(num_gpus=1)
    )
]

datasets = [
    dict(
        type='PIQADataset',
        abbr='piqa-dev',
        split='dev'
    ),
    dict(
        type='PIQADataset',
        abbr='piqa-test',
        split='test'
    )
]

work_dir = '/mnt/data/opencompass_results'
