python -u full-supervised.py --data cornell --layer 2 --weight_decay 1e-4 --model MLP --hidden 16
python -u full-supervised.py --data texas --layer 2 --weight_decay 1e-4 --model MLP --hidden 16
python -u full-supervised.py --data wisconsin --layer 2 --weight_decay 1e-4 --model MLP --hidden 16
python -u full-supervised.py --data chameleon --layer 2 --weight_decay 1e-8 --model MLP --hidden 16
python -u full-supervised.py --data cora --layer 2 --weight_decay 1e-4 --model MLP --hidden 16
python -u full-supervised.py --data citeseer --layer 2 --weight_decay 1e-7 --model MLP --hidden 64
python -u full-supervised.py --data film --layer 2 --weight_decay 1e-3 --model MLP --hidden 16 --dropout 0
python -u full-supervised.py --data squirrel --layer 2 --weight_decay 1e-4 --model MLP --hidden 16 --dropout 0.6
python -u full-supervised.py --data pubmed --layer 2 --weight_decay 1e-6 --model MLP --hidden 64 --dropout 0.3

