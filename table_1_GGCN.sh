python -u full-supervised.py --data wisconsin --layer 5 --weight_decay 6e-4 --model GGCN --hidden 16 --dropout 0
python -u full-supervised.py --data texas --layer 2 --weight_decay 1e-3 --model GGCN --hidden 16 --dropout 0.4
python -u full-supervised.py --data cora --layer 32 --weight_decay 1e-3 --model GGCN --hidden 16 --decay_rate 0.9
python -u full-supervised.py --data citeseer --layer 10 --weight_decay 1e-7 --model GGCN --hidden 80 --decay_rate 0.02
python -u full-supervised.py --data chameleon --layer 5 --weight_decay 1e-2 --model GGCN --hidden 32 --dropout 0.3 --decay_rate 0.8
python -u full-supervised.py --data cornell --layer 6 --weight_decay 1e-3 --model GGCN --hidden 16 --decay_rate 0.7
python -u full-supervised.py --data squirrel --layer 2 --weight_decay 4e-3 --model GGCN --hidden 32 --dropout 0.5
python -u full-supervised.py --data film --layer 4 --weight_decay 7e-3 --model GGCN --hidden 32 --dropout 0 --decay_rate 1.2
python -u full-supervised.py --data pubmed --layer 5 --weight_decay 1e-5 --model GGCN --hidden 32 --use_sparse --dropout 0.3 --decay_rate 1.1

