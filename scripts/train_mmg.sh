python3 train_mmg.py --train_dataset C/P/S --test_dataset W --missing depth --epochs 20 --batch_size 16 --lr 0.00015
python3 train_mmg.py --train_dataset C/P/W --test_dataset S --missing depth --epochs 20 --batch_size 16 --lr 0.00015
python3 train_mmg.py --train_dataset C/S/W --test_dataset P --missing depth --epochs 20 --batch_size 16 --lr 0.00015
python3 train_mmg.py --train_dataset P/S/W --test_dataset C --missing depth --epochs 20 --batch_size 16 --lr 0.00015

python3 train_mmg.py --train_dataset C/P/S --test_dataset W --missing ir --epochs 20 --batch_size 16 --lr 0.00015
python3 train_mmg.py --train_dataset C/P/W --test_dataset S --missing ir --epochs 20 --batch_size 16 --lr 0.00015
python3 train_mmg.py --train_dataset C/S/W --test_dataset P --missing ir --epochs 20 --batch_size 16 --lr 0.00015
python3 train_mmg.py --train_dataset P/S/W --test_dataset C --missing ir --epochs 20 --batch_size 16 --lr 0.00015