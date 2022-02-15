python -u full-supervised.py --data wisconsin --layer 2 --weight_decay 5e-5 --model GEOMGCN --hidden 32 --dropout 0.5 --nb_heads 1 --lr 0.05
python -u full-supervised.py --data texas --layer 2 --weight_decay 5e-5 --model GEOMGCN --hidden 32 --dropout 0.5 --nb_heads 1 --lr 0.05
python -u full-supervised.py --data squirrel --layer 2 --weight_decay 5e-6 --model GEOMGCN --hidden 48 --dropout 0.5 --nb_heads 1 --lr 0.05
python -u full-supervised.py --data film --layer 2 --weight_decay 5e-6 --model GEOMGCN --hidden 32 --dropout 0.5 --nb_heads 1 --lr 0.05
python -u full-supervised.py --data pubmed --layer 2 --weight_decay 5e-6 --model GEOMGCN --hidden 64 --dropout 0.5 --nb_heads 1 --lr 0.05 --emb MDS
python -u full-supervised.py --data cornell --layer 2--weight_decay 5e-5 --model GEOMGCN --hidden 32 --dropout 0.5 --nb_heads 1 --lr 0.05
python -u full-supervised.py --data chameleon --layer 2 --weight_decay 5e-6 --model GEOMGCN --hidden 48 --dropout 0.5 --nb_heads 1 --lr 0.05
python -u full-supervised.py --data cora --layer 2 --weight_decay 5e-6 --model GEOMGCN --hidden 16 --dropout 0.5 --nb_heads 1 --lr 0.05 --emb MDS
python -u full-supervised.py --data citeseer --layer 2 --weight_decay 5e-6 --model GEOMGCN --hidden 16 --dropout 0.5 --nb_heads 1 --lr 0.05 --emb MDS
