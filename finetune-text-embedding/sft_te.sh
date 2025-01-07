seed=1
data_path="/data/gumbou/codespace/work-backup/LRSL/multi-label/finetune-dataset/bert-aapd-train-semantic-top5.csv"
text_embedding="/data/public_model/text_embedding/bge-m3"
save_path="/data/gumbou/codespace/work-backup/LRSL/llms-cls-qlora/multi-label/te/bge"

CUDA_VISIBLE_DEVICES=2,3 python start_finetune.py $seed $data_path $te_path $save_path