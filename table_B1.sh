python -u full-supervised.py --data chameleon --layer 2 --weight_decay 6e-4 --model GGCN --hidden 32 --dropout 0.7 --no_decay --no_sign --no_degree
python -u full-supervised.py --data cora --layer 2 --weight_decay 1e-3 --model GGCN --hidden 64 --no_decay --no_sign --no_degree
python -u full-supervised.py --data cornell --layer 2 --weight_decay 6e-4 --model GGCN --hidden 8 --no_decay --no_sign --no_degree
python -u full-supervised.py --data citeseer --layer 2 --weight_decay 1e-7 --model GGCN --hidden 80 --no_decay --no_sign --no_degree
python -u full-supervised.py --data chameleon --layer 2 --weight_decay 6e-4 --model GGCN --hidden 32 --dropout 0.7 --deg_intercept_init 1.3 --no_sign --no_decay
python -u full-supervised.py --data cora --layer 2 --weight_decay 1e-3 --model GGCN --hidden 64 --deg_intercept_init 0.4 --no_sign  --no_decay
python -u full-supervised.py --data cornell --layer 2 --weight_decay 6e-4 --model GGCN --hidden 8 --deg_intercept_init 0.8 --no_sign  --no_decay
python -u full-supervised.py --data citeseer --layer 2 --weight_decay 1e-7 --model GGCN --hidden 80 --deg_intercept_init 0.2 --no_sign  --no_decay
python -u full-supervised.py --data chameleon --layer 2 --weight_decay 6e-4 --model GGCN --hidden 32 --dropout 0.7 --no_degree --scale_init 0.2 --no_decay
python -u full-supervised.py --data cora --layer 2 --weight_decay 1e-3 --model GGCN --hidden 64 --no_degree --scale_init 0.7  --no_decay
python -u full-supervised.py --data cornell --layer 2 --weight_decay 6e-4 --model GGCN --hidden 8 --no_degree --scale_init 0.2  --no_decay
python -u full-supervised.py --data citeseer --layer 2 --weight_decay 1e-7 --model GGCN --hidden 80 --no_degree --scale_init 1.0  --no_decay
python -u full-supervised.py --data chameleon --layer 2 --weight_decay 6e-4 --model GGCN --hidden 32 --dropout 0.7 --deg_intercept_init 1.3 --scale_init 0.2 --no_decay
python -u full-supervised.py --data cora --layer 2 --weight_decay 1e-3 --model GGCN --hidden 64 --deg_intercept_init 0.4 --scale_init 0.7  --no_decay
python -u full-supervised.py --data cornell --layer 2 --weight_decay 6e-4 --model GGCN --hidden 8 --deg_intercept_init 0.8 --scale_init 0.2  --no_decay
python -u full-supervised.py --data citeseer --layer 2 --weight_decay 1e-7 --model GGCN --hidden 80 --deg_intercept_init 0.2 --scale_init 1.0  --no_decay

