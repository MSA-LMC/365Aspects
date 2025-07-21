# Define experiment name as a variable
EXPERIMENT_NAME="202507010_1"

nohup python -u train_task2_vote.py --loss_plot_path ./loss_img/loss_curve_${EXPERIMENT_NAME}.png \
	--output_model best_model_${EXPERIMENT_NAME}.pth \
	--log_dir ./log \
	--training_time 2025-06-15 \
	--train_csv ./data/train_data.csv \
	--val_csv ./data/val_data_new.csv \
	--test_csv ./data/test_data_basic_information.csv \
	--question q1 q2 q3 q4 q5 q6 \
	--label_col Integrity Collegiality Social_versatility Development_orientation Hireability \
	--rating_csv ./data/all_data.csv \
	--video_dim 1152 \
	--video_dir /home/gdp/AVI/data/face_embedding/siglip2_all_maxP_face \
	--audio_dim 768 \
	--audio_dir /home/gdp/AVI/data/audioFeatures/audioFeatures/train/emotion2vec_plus_seed \
	--text_dim 4096 \
	--text_dir /home/gdp/AVI/data/text_feature/SFR-Embedding-Mistral \
	--batch_size 64 \
	--learning_rate 1e-4 \
	--num_epochs 200 \
	--test_output_csv ./output_${EXPERIMENT_NAME}.csv \
	--test_model best_model_20250624_2_copy.pth \
	> ./train_print_log/${EXPERIMENT_NAME}.log 2>&1 &
