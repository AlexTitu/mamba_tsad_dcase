datasets="NASA SMD SWaT"
# datasets="NASA"
tag="final_trial"
for data in ${datasets}; do
  echo ${data}
  nohup python3 main.py \
  --dataset ${data} \
  --decomp \
  --pre_filter \
  --tag ${tag} \
  > run_${data}_${tag}.log 2>&1 &
done





