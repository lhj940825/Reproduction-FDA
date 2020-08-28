python3 train.py --snapshot-dir='../checkpoints/FDA' --init-weights='../checkpoints/FDA/init_weight/DeepLab_init.pth' --LB=0.01 --entW=0.005 --ita=2.0 --switch2entropy=0
python3 train.py --snapshot-dir='../checkpoints/FDA' --init-weights='../checkpoints/FDA/init_weight/DeepLab_init.pth' --LB=0.05 --entW=0.005 --ita=2.0 --switch2entropy=0
python3 train.py --snapshot-dir='../checkpoints/FDA' --init-weights='../checkpoints/FDA/init_weight/DeepLab_init.pth' --LB=0.09 --entW=0.005 --ita=2.0 --switch2entropy=0
