# python exec.py preprocess_tum /mnt/data/teamAI/duy/data/TUM/dataset-corridor1_512_16/mav0 /mnt/data/teamAI/duy/deep_ekf_vio/data 0 40
python preprocess/associate.py /mnt/data/teamAI/duy/data/TUM/dataset-corridor1_512_16/mav0/imu0/data.csv\
                                 /mnt/data/teamAI/duy/data/TUM/dataset-corridor1_512_16/mav0/mocap0/data.csv\
                                 --max_difference 1000
