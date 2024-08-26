#!/usr/bin/env bash

# TAPORUN="False"
# echo $TAPORUN
# echo 'here'

cmd1=$1
cmd2=$2
# cmd3="python TAPO.py ${cmd2}"


# ${cmd1} 2>&1 | tee $cmd2 &  TASK=$!
${cmd1} &  TASK=$!
python TAPO.py $cmd2 &  QUERY=$!
wait $TASK 
wait $QUERY



# Previous commands directly inside experiments.sh:
    # ${cmd} &  TASK=$!
    # python TAPO.py ${opt_path} &  QUERY=$!
    # wait $TASK 
    # wait $QUERY
    # 2>&1 | tee ${log_file}.txt