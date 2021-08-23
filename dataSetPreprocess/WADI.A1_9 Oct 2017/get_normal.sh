tail -n +5 WADI_14days.csv > WADI_normal.csv
python label_normal.py
head -n 1 WADI_normal_2.csv > WADI_normal_pre.csv
tail -n 500000 WADI_normal_2.csv >> WADI_normal_pre.csv
cat WADI_normal_pre.csv | cut -d',' -f2- > WADI_normal_pre_2.csv
