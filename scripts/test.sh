python3 test.py --train_dataset C/P/S --test_dataset W --pmc True
python3 test.py --train_dataset C/P/W --test_dataset S --pmc True
python3 test.py --train_dataset C/S/W --test_dataset P --pmc True
python3 test.py --train_dataset P/S/W --test_dataset C --pmc True

python3 test.py --train_dataset C/P/S --test_dataset W --missing depth --model_save_step 200
python3 test.py --train_dataset C/P/W --test_dataset S --missing depth --model_save_step 200
python3 test.py --train_dataset C/S/W --test_dataset P --missing depth --model_save_step 200
python3 test.py --train_dataset P/S/W --test_dataset C --missing depth --model_save_step 200

python3 test.py --train_dataset C/P/S --test_dataset W --missing ir --model_save_step 200
python3 test.py --train_dataset C/P/W --test_dataset S --missing ir --model_save_step 200
python3 test.py --train_dataset C/S/W --test_dataset P --missing ir --model_save_step 200
python3 test.py --train_dataset P/S/W --test_dataset C --missing ir --model_save_step 200

python3 test.py --train_dataset C/P/S --test_dataset W --missing depth+ir --model_save_step 400
python3 test.py --train_dataset C/P/W --test_dataset S --missing depth+ir --model_save_step 400
python3 test.py --train_dataset C/S/W --test_dataset P --missing depth+ir --model_save_step 400
python3 test.py --train_dataset P/S/W --test_dataset C --missing depth+ir --model_save_step 400