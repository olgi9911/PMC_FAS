python3 adapt.py --train_dataset C/P/S --test_dataset W --epochs 10 --batch_size 16 --lr 0.0015
python3 adapt.py --train_dataset C/P/W --test_dataset S --epochs 10 --batch_size 16 --lr 0.0015
python3 adapt.py --train_dataset C/S/W --test_dataset P --epochs 10 --batch_size 16 --lr 0.0015
python3 adapt.py --train_dataset P/S/W --test_dataset C --epochs 10 --batch_size 16 --lr 0.0015