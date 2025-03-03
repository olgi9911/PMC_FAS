python3 adapt_missing.py --train_dataset C/P/S --test_dataset W --missing depth --epochs 5 --batch_size 16 --lr 0.00015 --model_save_step 200
python3 adapt_missing.py --train_dataset C/P/W --test_dataset S --missing depth --epochs 5 --batch_size 16 --lr 0.00015 --model_save_step 200
python3 adapt_missing.py --train_dataset C/S/W --test_dataset P --missing depth --epochs 5 --batch_size 16 --lr 0.00015 --model_save_step 200
python3 adapt_missing.py --train_dataset P/S/W --test_dataset C --missing depth --epochs 5 --batch_size 16 --lr 0.00015 --model_save_step 200

python3 adapt_missing.py --train_dataset C/P/S --test_dataset W --missing ir --epochs 5 --batch_size 16 --lr 0.00015 --model_save_step 200
python3 adapt_missing.py --train_dataset C/P/W --test_dataset S --missing ir --epochs 5 --batch_size 16 --lr 0.00015 --model_save_step 200
python3 adapt_missing.py --train_dataset C/S/W --test_dataset P --missing ir --epochs 5 --batch_size 16 --lr 0.00015 --model_save_step 200
python3 adapt_missing.py --train_dataset P/S/W --test_dataset C --missing ir --epochs 5 --batch_size 16 --lr 0.00015 --model_save_step 200

python3 adapt_missing.py --train_dataset C/P/S --test_dataset W --missing depth+ir --epochs 5 --batch_size 8 --lr 0.00015 --model_save_step 400
python3 adapt_missing.py --train_dataset C/P/W --test_dataset S --missing depth+ir --epochs 5 --batch_size 8 --lr 0.00015 --model_save_step 400
python3 adapt_missing.py --train_dataset C/S/W --test_dataset P --missing depth+ir --epochs 5 --batch_size 8 --lr 0.00015 --model_save_step 400
python3 adapt_missing.py --train_dataset P/S/W --test_dataset C --missing depth+ir --epochs 5 --batch_size 8 --lr 0.00015 --model_save_step 400