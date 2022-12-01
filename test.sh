



seeds='13 27 250 583 915'

for seed in $seeds; do

echo '

' $seed '
'

python run_PALP.py seed=$seed

done