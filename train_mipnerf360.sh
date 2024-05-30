exp_name="baseline"
gpu=-1
ratio=1
resolution=-1
appearance_dim=0

fork=2
base_layer=-1
visible_threshold=-1 #0.9 
dist2level="round"
update_ratio=0.2

progressive="True"
dist_ratio=0.999 #0.99
levels=-1
init_level=-1
extra_ratio=0.25
extra_up=0.01

# example:
./train.sh -d 'mipnerf360/bicycle' -l ${exp_name} --gpu ${gpu} -r ${resolution} --ratio ${ratio} --appearance_dim ${appearance_dim} \
   --fork ${fork} --visible_threshold ${visible_threshold} --base_layer ${base_layer} --dist2level ${dist2level} --update_ratio ${update_ratio} \
   --progressive ${progressive} --levels ${levels} --init_level ${init_level}  --dist_ratio ${dist_ratio} \
   --extra_ratio ${extra_ratio} --extra_up ${extra_up} &
sleep 20s

./train.sh -d 'mipnerf360/garden' -l ${exp_name} --gpu ${gpu} -r ${resolution} --ratio ${ratio} --appearance_dim ${appearance_dim} \
   --fork ${fork} --visible_threshold ${visible_threshold} --base_layer ${base_layer} --dist2level ${dist2level} --update_ratio ${update_ratio} \
   --progressive ${progressive} --levels ${levels} --init_level ${init_level}  --dist_ratio ${dist_ratio} \
   --extra_ratio ${extra_ratio} --extra_up ${extra_up} &
sleep 20s

./train.sh -d 'mipnerf360/stump' -l ${exp_name} --gpu ${gpu} -r ${resolution} --ratio ${ratio} --appearance_dim ${appearance_dim} \
   --fork ${fork} --visible_threshold ${visible_threshold} --base_layer ${base_layer} --dist2level ${dist2level} --update_ratio ${update_ratio} \
   --progressive ${progressive} --levels ${levels} --init_level ${init_level}  --dist_ratio ${dist_ratio} \
   --extra_ratio ${extra_ratio} --extra_up ${extra_up} &
sleep 20s

./train.sh -d 'mipnerf360/room' -l ${exp_name} --gpu ${gpu} -r ${resolution} --ratio ${ratio} --appearance_dim ${appearance_dim} \
   --fork ${fork} --visible_threshold ${visible_threshold} --base_layer ${base_layer} --dist2level ${dist2level} --update_ratio ${update_ratio} \
   --progressive ${progressive} --levels ${levels} --init_level ${init_level}  --dist_ratio ${dist_ratio} \
   --extra_ratio ${extra_ratio} --extra_up ${extra_up} &
sleep 20s

./train.sh -d 'mipnerf360/counter' -l ${exp_name} --gpu ${gpu} -r ${resolution} --ratio ${ratio} --appearance_dim ${appearance_dim} \
   --fork ${fork} --visible_threshold ${visible_threshold} --base_layer ${base_layer} --dist2level ${dist2level} --update_ratio ${update_ratio} \
   --progressive ${progressive} --levels ${levels} --init_level ${init_level}  --dist_ratio ${dist_ratio} \
   --extra_ratio ${extra_ratio} --extra_up ${extra_up} &
sleep 20s

./train.sh -d 'mipnerf360/kitchen' -l ${exp_name} --gpu ${gpu} -r ${resolution} --ratio ${ratio} --appearance_dim ${appearance_dim} \
   --fork ${fork} --visible_threshold ${visible_threshold} --base_layer ${base_layer} --dist2level ${dist2level} --update_ratio ${update_ratio} \
   --progressive ${progressive} --levels ${levels} --init_level ${init_level}  --dist_ratio ${dist_ratio} \
   --extra_ratio ${extra_ratio} --extra_up ${extra_up} &
sleep 20s

./train.sh -d 'mipnerf360/bonsai' -l ${exp_name} --gpu ${gpu} -r ${resolution} --ratio ${ratio} --appearance_dim ${appearance_dim} \
   --fork ${fork} --visible_threshold ${visible_threshold} --base_layer ${base_layer} --dist2level ${dist2level} --update_ratio ${update_ratio} \
   --progressive ${progressive} --levels ${levels} --init_level ${init_level}  --dist_ratio ${dist_ratio} \
   --extra_ratio ${extra_ratio} --extra_up ${extra_up} &
sleep 20s

./train.sh -d 'mipnerf360/flowers' -l ${exp_name} --gpu ${gpu} -r ${resolution} --ratio ${ratio} --appearance_dim ${appearance_dim} \
   --fork ${fork} --visible_threshold ${visible_threshold} --base_layer ${base_layer} --dist2level ${dist2level} --update_ratio ${update_ratio} \
   --progressive ${progressive} --levels ${levels} --init_level ${init_level}  --dist_ratio ${dist_ratio} \
   --extra_ratio ${extra_ratio} --extra_up ${extra_up} &
sleep 20s

./train.sh -d 'mipnerf360/treehill' -l ${exp_name} --gpu ${gpu} -r ${resolution} --ratio ${ratio} --appearance_dim ${appearance_dim} \
   --fork ${fork} --visible_threshold ${visible_threshold} --base_layer ${base_layer} --dist2level ${dist2level} --update_ratio ${update_ratio} \
   --progressive ${progressive} --levels ${levels} --init_level ${init_level}  --dist_ratio ${dist_ratio} \
   --extra_ratio ${extra_ratio} --extra_up ${extra_up} &