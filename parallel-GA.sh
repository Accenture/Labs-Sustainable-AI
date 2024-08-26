#!/usr/bin/env bash


# TODO

cmd1=$1
cmd2=$2
echo $cmd1
echo $cmd2


# ${cmd1} 2>&1 | tee $cmd2 &  TASK=$!
${cmd1} &  TASK=$!
python UTIL.py $cmd2 &  QUERY=$!
wait $TASK 
wait $QUERY



# Previous commands directly inside experiments.sh:
    # ${cmd} &  TASK=$!
    # python UTIL.py ${opt_path} &  QUERY=$!
    # wait $TASK 
    # wait $QUERY
    # 2>&1 | tee ${log_file}.txt