function rand(){
    min=$1
    max=$(($2-$min+1))
    num=$(date +%s%N)
    echo $(($num%$max+$min))  
}

port=$(rand 10000 30000)

iterations=40_000
warmup="False"
progressive="False"
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -l|--logdir) logdir="$2"; shift ;;
        -d|--data) data="$2"; shift ;;
        -r|--resolution) resolution="$2"; shift ;;
        --gpu) gpu="$2"; shift ;;
        --ratio) ratio="$2"; shift ;;
        --warmup) warmup="$2"; shift ;;
        --appearance_dim) appearance_dim="$2"; shift ;;
        --fork) fork="$2"; shift ;;
        --base_layer) base_layer="$2"; shift ;;
        --visible_threshold ) visible_threshold="$2"; shift ;;
        --dist2level) dist2level="$2"; shift ;;
        --update_ratio) update_ratio="$2"; shift ;;
        --progressive) progressive="$2"; shift ;;
        --levels) levels="$2"; shift ;;
        --dist_ratio) dist_ratio="$2"; shift ;;
        --init_level) init_level="$2"; shift ;;
        --extra_ratio) extra_ratio="$2"; shift ;;
        --extra_up) extra_up="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

time=$(date "+%Y-%m-%d_%H:%M:%S")

if [ "$progressive" = "True" ]; then
    if [ "$warmup" = "True" ]; then
        python train.py --eval -s data/${data} -r ${resolution} --gpu ${gpu} --fork ${fork} --ratio ${ratio} --warmup \
        --iterations ${iterations} --port $port -m outputs/${data}/${logdir}/$time --appearance_dim ${appearance_dim} \
        --visible_threshold ${visible_threshold}  --base_layer ${base_layer} --dist2level ${dist2level} --update_ratio ${update_ratio} \
        --progressive --init_level ${init_level} --dist_ratio ${dist_ratio} --levels ${levels}  \
        --extra_ratio ${extra_ratio} --extra_up ${extra_up}
    else
        python train.py --eval -s data/${data} -r ${resolution} --gpu ${gpu} --fork ${fork} --ratio ${ratio} \
        --iterations ${iterations} --port $port -m outputs/${data}/${logdir}/$time --appearance_dim ${appearance_dim} \
        --visible_threshold ${visible_threshold}  --base_layer ${base_layer} --dist2level ${dist2level} --update_ratio ${update_ratio} \
        --progressive --init_level ${init_level} --dist_ratio ${dist_ratio} --levels ${levels}  \
        --extra_ratio ${extra_ratio} --extra_up ${extra_up}  
    fi
else
    if [ "$warmup" = "True" ]; then
        python train.py --eval -s data/${data} -r ${resolution} --gpu ${gpu} --fork ${fork} --ratio ${ratio} --warmup \
        --iterations ${iterations} --port $port -m outputs/${data}/${logdir}/$time --appearance_dim ${appearance_dim} \
        --visible_threshold ${visible_threshold}  --base_layer ${base_layer} --dist2level ${dist2level} --update_ratio ${update_ratio} \
        --init_level ${init_level} --dist_ratio ${dist_ratio} --levels ${levels}  \
        --extra_ratio ${extra_ratio} --extra_up ${extra_up}  
    else
        python train.py --eval -s data/${data} -r ${resolution} --gpu ${gpu} --fork ${fork} --ratio ${ratio} \
        --iterations ${iterations} --port $port -m outputs/${data}/${logdir}/$time --appearance_dim ${appearance_dim} \
        --visible_threshold ${visible_threshold}  --base_layer ${base_layer} --dist2level ${dist2level} --update_ratio ${update_ratio} \
        --init_level ${init_level} --dist_ratio ${dist_ratio} --levels ${levels}  \
        --extra_ratio ${extra_ratio} --extra_up ${extra_up}  
    fi
fi
