使用：
python train.py --data_csv ./cry_data/data.csv --model_type rf --output_path models/cry_detection_model_rf.joblib <br/>
python inference.py --model_path models/cry_detection_model_rf.joblib --input_path ./test_data/test.wav --output_dir results <br/>
