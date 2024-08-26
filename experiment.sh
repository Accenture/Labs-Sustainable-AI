#!/usr/bin/env bash


# ------------------- SET BY THE USER ----------------------- #

# indicate which folders contain the data for the training:
export BERT_BASE_DIR=/home/demouser/Documents/Demos/energycalculatorsevaluation/data/bert/uncased_L-12_H-768_A-12 
export SQUAD_DIR=/home/demouser/Documents/Demos/energycalculatorsevaluation/data/bert/data 
export IMAGE_NET_DIR=/media/demouser/0434B71B34B70F24/image_net

# choose which evaluation tools or methods you want to include in the experiments
calc_list=(
'code_carbon:online' \
'carbon_tracker:measure' \
'carbon_tracker:predict' \
'eco2ai' \
'green_algorithms:default' \
'green_algorithms:automated_parallel' \
'tapo'
)
calc_list=(
'green_algorithms:default'
)

# choose if you want to use a GPU for the training:
accelerator_list=('True')

# choose the number of iterations you want to do during the experiments:
nbIter=2

# here change the exp name (image_net, SQUAD-v1-1, SQUAD-extracted, idle, mnist, etc):
exp_list=('image_net')

# if dev_test is "True", less epochs will be used for the training (for tests):
dev_test="True"

# ------------------- SET BY THE USER ----------------------- #




# ----------------- UNCOMMENT FOR EXPERIMENTS -------------------- #

sudo chmod o+r /sys/class/powercap/intel-rapl\:0/energy_uj

sudo chmod a+r /dev/cpu/*/msr
sudo setcap cap_sys_rawio=ep /usr/sbin/rdmsr

# Needed ?
# export OUTPUT_ESTIMATOR=./output/calculator_output 

# ----------------- UNCOMMENT FOR EXPERIMENTS -------------------- #



# ------------------- USEFULL VAR AND FCTS ----------------------- #

declare -A scripts=( ['mnist']='exp-1-MNIST/MNIST-ALL' \
['cifar10']='exp-2-CIFAR10/CIFAR10-ALL' \
['CUB_200_2011']='exp-3-resnet18/classification_pytorch_vision' \
['image_net']='exp-3-resnet18/classification_pytorch_vision' \
['SQUAD-extracted']='exp-4-bert-squad/run_squad' \
['SQUAD-v1-1']='exp-4-bert-squad/run_squad' \
['idle']='exp-0-idle/idle' )

declare -A acc_list=( [False]='cpu' [True]='cuda')

declare -A squad_data=( ["training"]="train" \
["inference"]="dev" \
["SQUAD-extracted"]="extracted" \
["SQUAD-v1-1"]="v1.1" )  

python initialisation_experiment.py 

path_logs_and_results=$(cat logs_and_results/current_logs_and_results_folder.txt)
# x-terminal-emulator -e "python TAPO-VAR-initialisation.py" # if we want each ml task to open in a different terminal
echo $path_logs_and_results


one_monitoring_step () {
    # Function body

    echo $calc_and_mode

    calc=$(echo $calc_and_mode | cut -d':' -f1)
    mode=$(echo $calc_and_mode | cut -d':' -f2)

    script=${scripts[$exp]}
    acc=${acc_list[$useAcc]}
    echo $exp $folder $i $calc $mode $acc
    file="${script}.py"


    opt1="--use_accelerator=${useAcc}"
    opt2="--save_model=False"
    opt3="--calculator=${calc}"
    opt4="--calculator_mode=${mode}"
    opt5="--ml_phase=${ml}"
    opt6="--dev_test=${dev_test}"
    opt7="--name_exp=${exp}"
    opt="${opt1} ${opt2} ${opt3} ${opt4} ${opt5} ${opt6} ${opt7}"

    if [[ $exp == 'mnist' ]]; then
        if [[ $dev_test == 'True' ]]; then
            opt7="--nb_batch_inferences=10"
            opt8="--epochs=2"
            opt="${opt} ${opt7} ${opt8}"
        fi
        if [[ $ml == 'training' ]]; then
            mkdir "${path_logs_and_results}/${i}/${exp}_model"
            opt9="--output_dir=${path_logs_and_results}/${i}/${exp}_model"
            opt="${opt} ${opt9}"
        fi
    fi

    if [[ $exp == 'cifar10' ]]; then
        if [[ $dev_test == 'True' ]]; then
            opt7="--nb_batch_inferences=50"
            opt8="--epochs=2"
            opt="${opt} ${opt7} ${opt8}"
        fi
        if [[ $ml == 'training' ]]; then
            mkdir "${path_logs_and_results}/${i}/${exp}_model"
            opt9="--output_dir=${path_logs_and_results}/${i}/${exp}_model"
            opt="${opt} ${opt9}"
        fi
    fi

    if [[ $exp == 'CUB_200_2011' ||  $exp == 'image_net' ]]; then
        if [[ $dev_test == 'True' ]]; then
            opt7="--nb_batch_inferences=5"
            opt8="--epochs=1"
            opt="${opt} ${opt7} ${opt8}"
        else
            opt7="--nb_batch_inferences=5"
            opt8="--epochs=2" 
            opt="${opt} ${opt7} ${opt8}"
        fi
        if [[ $exp == 'image_net' ]]; then
            opt9="--data-path=${IMAGE_NET_DIR}"
        else
            opt9="--data-path=data/${exp}"
        fi
        if [[ $ml == 'training' ]]; then
            mkdir "${path_logs_and_results}/${i}/${exp}_model"
            opt10="--output-dir=${path_logs_and_results}/${i}/${exp}_model"
        fi
        opt="${opt} ${opt9} ${opt10}"

        o1="--batch-size=16 --workers=8"
        opt="${opt} ${o1} ${o2} ${o3} ${o4} ${o5} ${o6} ${o7}"
    fi

    if [[ $exp == "SQUAD-extracted" || $exp == "SQUAD-v1-1" ]]; then
        file_name="$SQUAD_DIR/train-${squad_data[$exp]}"
        opt7="--train_file=${file_name}.json"
        file_name="$SQUAD_DIR/dev-${squad_data[$exp]}"
        opt8="--predict_file=${file_name}.json"

        if [[ $ml == "training" ]]; then
            opt9="--do_predict=False --do_train=True"
            mkdir "${path_logs_and_results}/${i}/${exp}_model"
            opt10="--output_dir=${path_logs_and_results}/${i}/${exp}_model"
        else
            opt9="--do_predict=True --do_train=False"
            mkdir "${path_logs_and_results}/{$i}/${exp}_inference"
            opt10="--output_dir=${path_logs_and_results}/${i}/${exp}_inference"
        fi
        opt="${opt} ${opt7} ${opt8} ${opt9} ${opt10}"
        if [[ $dev_test == True ]]; then
            opt11="--num_train_epochs=1.0"
        else
            opt11="--num_train_epochs=2.0"
        fi
        opt="${opt} ${opt11} ${bert_opt}"

        o1="--train_batch_size=8"
        o2="--learning_rate=3e-5"
        o3="--vocab_file=$BERT_BASE_DIR/vocab.txt"
        o4="--bert_config_file=$BERT_BASE_DIR/bert_config.json"
        o5="--init_checkpoint=${BERT_BASE_DIR}/bert_model.ckpt"
        o6="--doc_stride=128"
        o7="--max_seq_length=128"
        opt="${opt} ${o1} ${o2} ${o3} ${o4} ${o5} ${o6} ${o7}"
    fi
    opt="${opt} ${opt_path}"
    }

python experiment_completed.py --done=False

ml="training"

# ------------------- USEFULL VAR AND FCTS ----------------------- #







# ------------------- EXPERIMENT LOOP ----------------------- #

if [[ $1 != "" ]]; then
    exp_list=($1)
    calc_list=('no_calculator')
    accelerator_list=('True')
    nbIter=1
    ml="training"
    dev_test="True" 
fi

idle_time=600

for i in `seq $nbIter`; do

    mkdir ${path_logs_and_results}/${i}
    opt_path="--path_logs_and_results=${path_logs_and_results}/${i}"
    mkdir ${path_logs_and_results}/${i}/carbon_tracker_measure_logs
    mkdir ${path_logs_and_results}/${i}/carbon_tracker_predict_logs
    mkdir ${path_logs_and_results}/${i}/tapo_logs
    mkdir ${path_logs_and_results}/${i}/term_logs
    mkdir ${path_logs_and_results}/${i}/util_logs


    for exp in ${exp_list[@]}; do 

        for useAcc in ${accelerator_list[@]}; do 
            shuf_calc_list=$(shuf -e "${calc_list[@]}")

            cpt=0
            for calc_and_mode in ${shuf_calc_list[@]}; do

                one_monitoring_step

                if [[ $exp == 'idle' ]]; then
                    cmd_idle="--idle_time=${idle_time}"
                fi

                cmd="python ${file} ${opt} ${cmd_idle}"
                echo $cmd
                log_file="${path_logs_and_results}/${i}/term_logs/${i}-${cpt}-${exp}-${acc}-${calc}-${mode}"

                echo $calc
                if [[ $calc == 'tapo' ]]; then
                    # x-terminal-emulator -e "./parallel-TAPO.sh '${cmd}' '${opt_path}' 2>&1 | tee ${log_file}.txt" # if we want each ml task to open in a different terminal
                    ./parallel-TAPO.sh "${cmd}" "${opt_path}" 2>&1 | tee ${log_file}.txt
                elif [[ $calc == 'green_algorithms' && $mode == 'automated_parallel' ]]; then
                    # x-terminal-emulator -e "./parallel-GA.sh '${cmd}' '${opt_path}' 2>&1 | tee ${log_file}.txt" # if we want each ml task to open in a different terminal
                    ./parallel-GA.sh "${cmd}" "${opt_path}" 2>&1 | tee ${log_file}.txt
                else
                    # x-terminal-emulator -e "${cmd} 2>&1 | tee ${log_file}.txt" # if we want each ml task to open in a different terminal
                    ${cmd} 2>&1 | tee ${log_file}.txt
                fi

                cpt=$((cpt+1))

                if [[ $exp == 'idle' ]]; then
                    sleep 2m
                else
                    if [[ $dev_test == 'True' ]]; then
                        sleep 10s
                    else
                        sleep 10m
                    fi
                fi               
            done

            if [[ $i == 1 && $exp != 'idle' ]]; then
                calc_and_mode='flops'
                one_monitoring_step
                cmd="python ${file} ${opt}"
                echo $cmd
                log_file="${path_logs_and_results}/${i}/term_logs/${i}-${cpt}-${exp}-${acc}-flops-${mode}"
                echo $calc
                # x-terminal-emulator -e "${cmd} 2>&1 | tee ${log_file}.txt" # if we want each ml task to open in a different terminal
                ${cmd} 2>&1 | tee ${log_file}.txt
            fi
            
        done
    done
done

python experiment_completed.py --done=True

# ------------------- EXPERIMENT LOOP ----------------------- #