for l in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5
do
python -u full-supervised.py --data chameleon --layer 4 --weight_decay 6e-4 --model GGCN --hidden 32 --dropout 0.7 --no_sign --no_degree --decay_rate $l > GGCN_chameleon_decay_$l.txt 2>&1
python -u full-supervised.py --data cora --layer 4 --weight_decay 1e-3 --model GGCN --hidden 64 --no_sign --no_degree --decay_rate --decay_rate $l > GGCN_cora_decay_$l.txt 2>&1
python -u full-supervised.py --data cornell --layer 4 --weight_decay 6e-4 --model GGCN --hidden 8 --no_sign --no_degree --decay_rate --decay_rate $l > GGCN_cornell_decay_$l.txt 2>&1
python -u full-supervised.py --data citeseer --layer 4 --weight_decay 1e-7 --model GGCN --hidden 80 --no_sign --no_degree --decay_rate --decay_rate $l > GGCN_citeseer_decay_$l.txt 2>&1
done

