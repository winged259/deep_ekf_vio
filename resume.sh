folder='/mnt/data/teamAI/duy/deep_ekf_vio/results/train_20230530-17-59-50'

python main.py --gpu_id=0\
                --resume_model_from $folder/saved_model.train\
                # --resume_optimizer_from $folder/saved_optimizer.checkpoint