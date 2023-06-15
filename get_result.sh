folder='/mnt/data/teamAI/duy/deep_ekf_vio/results/new_save_untested'
model=$folder/saved_model.train

python exec.py gen_trajectory $model

python exec.py plot_trajectory $folder/saved_model.train.traj

python exec.py calc_error  $folder/saved_model.train.traj

python exec.py plot_error $folder/saved_model.train.traj

python exec.py np_traj_to_kitti  $folder/saved_model.train.traj

python exec.py kitti_eval $folder/saved_model.train.traj