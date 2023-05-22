folder='/mnt/data/teamAI/duy/deep_ekf_vio/results/train_20230520-02-55-40'

python main.py --gpu_id=0\
                --resume_model_from $folder/saved_model.train\
                # --resume_optimizer_from $folder/saved_optimizer.checkpoint