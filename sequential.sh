# python3 robustness_check/train_with_c_driving.py --LB=0.01 --entW=0.005 --ita=2.0 --switch2entropy=0 --FDA_mode='on' --weather='cloudy'
# python3 robustness_check/train_with_c_driving.py --LB=0.01 --entW=0.005 --ita=2.0 --switch2entropy=0 --FDA_mode='off' --weather='cloudy'
# python3 robustness_check/train_with_c_driving.py --LB=0.01 --entW=0.005 --ita=2.0 --switch2entropy=0 --FDA_mode='on' --weather='rainy'
# python3 robustness_check/train_with_c_driving.py --LB=0.01 --entW=0.005 --ita=2.0 --switch2entropy=0 --FDA_mode='off' --weather='rainy'


python3 robustness_check/train_with_c_driving.py --LB=0.01 --entW=0.005 --ita=2.0 --switch2entropy=0 --FDA_mode='off' --weather='rainy'
python3 robustness_check/train_with_c_driving.py --LB=0.01 --entW=0.005 --ita=2.0 --switch2entropy=0 --FDA_mode='on' --weather='snowy'
python3 robustness_check/train_with_c_driving.py --LB=0.01 --entW=0.005 --ita=2.0 --switch2entropy=0 --FDA_mode='off' --weather='snowy'
python3 robustness_check/train_with_c_driving.py --LB=0.01 --entW=0.005 --ita=2.0 --switch2entropy=0 --FDA_mode='off' --weather='overcast' --_type='open'
python3 robustness_check/train_with_c_driving.py --LB=0.01 --entW=0.005 --ita=2.0 --switch2entropy=0 --FDA_mode='on' --weather='overcast' --_type='open'