folder='/mnt/data/teamAI/duy/deep_ekf_vio/results/train_20230504-17-20-25'

python main.py --gpu_id=0\
                --resume_model_from $folder/saved_model.valid\
                --resume_optimizer_from $folder/saved_optimizer.checkpoint