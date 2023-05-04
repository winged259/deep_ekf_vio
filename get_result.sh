folder='/mnt/data/teamAI/duy/deep_ekf_vio/results/train_20230501-08-35-36'
model=$folder/saved_model.train

python exec.py gen_trajectory $model

python exec.py plot_trajectory $folder/saved_model.train.traj
