import sys
import os
_path = '.'
sys.path.append(os.path.join(_path))
import json
import time

# --- FOR CALCULATORS
from codecarbon import EmissionsTracker, OfflineEmissionsTracker
from carbontracker.tracker import CarbonTracker
import eco2ai
from fct_for_ga import wait_for_UTIL
from fct_for_tapo import wait_for_TAPO
# ------------------


def new_timestamp(main_folder, iteration, tag):

    file_name = os.path.join(main_folder, 'timestamps.json')

    with open(file_name, 'r') as file:
        d = json.load(file)

    if iteration not in d:
        d[iteration] = {}
    d[iteration][tag] = time.time()

    with open(file_name, 'w') as file:
        json.dump(d, file, indent = 4)


class ExpParams():
    def __init__(self, args_parser):

        self.name = args_parser.name_exp
        self.comp = args_parser.computer
        self.name_calc = args_parser.calculator
        self.mode_calc = args_parser.calculator_mode if args_parser.calculator_mode != self.name_calc else ''
        self.path_logs_and_results = args_parser.path_logs_and_results

        tmp = self.path_logs_and_results
        tmp = tmp.split("/")
        self.main_folder = "/".join(tmp[:-1])
        self.iteration = tmp[-1]

        # timestamp - calculator iteration test start
        tag = self.name_calc + ':' + self.mode_calc + ' test start'
        new_timestamp(self.main_folder, self.iteration, tag)

        if self.name[:5] == "SQUAD":
            import tensorflow as tf
            self.use_accelerator = args_parser.use_accelerator
            self.save_model = False
            if args_parser.do_predict == True:
                self.ml = "inference"
                self.epochs = "N/A"
            else: # we don't do both
                self.ml = "training"
                self.epochs = args_parser.num_train_epochs
            self.dev_test = None
            self.train_batch_size = args_parser.train_batch_size
            self.test_batch_size = args_parser.predict_batch_size
        else:
            import torch
            self.use_accelerator = bool(args_parser.use_accelerator == "True")
            self.save_model = bool(args_parser.save_model == "True")
            self.ml = args_parser.ml_phase
            if self.ml == "inference":
                self.epochs = "N/A"
            else:
                self.epochs = args_parser.epochs
            self.dev_test = bool(args_parser.dev_test == "True")
            if (self.name == 'cifar10') or (self.name == 'mnist'):
                self.train_batch_size = args_parser.batch_size
                self.test_batch_size = args_parser.test_batch_size
            else:
                self.train_batch_size = args_parser.batch_size
                self.test_batch_size = args_parser.batch_size



        print('Experience: ', self.name)
        print('Save model: ', self.save_model)
        print('Accelerator: ', self.use_accelerator)
        print('ML phase: ', self.ml)
        print('Calculator: ', self.name_calc)
        print('Dev test model: ', self.dev_test)

        self.measure = None
        self.online = None
        self.automated = None
        self.parallel = None

        if self.name_calc == 'carbon_tracker':
            self.measure = bool(args_parser.calculator_mode != "predict")
            print('Calculator mode: ', args_parser.calculator_mode, ' measure=', self.measure)
            if self.measure:
                self.ct_log_dir = os.path.join(args_parser.path_logs_and_results, "carbon_tracker_measure_logs")
            else:
                self.ct_log_dir = os.path.join(args_parser.path_logs_and_results, "carbon_tracker_predict_logs")
        elif self.name_calc == 'code_carbon':
            self.online = bool(args_parser.calculator_mode != "offline")
            print('Calculator mode: ', args_parser.calculator_mode, ' online=', self.online)
            if self.online:
                self.cc_output_file = os.path.join(args_parser.path_logs_and_results, "output_code_carbon_online.csv")
            else:
                self.cc_output_file = os.path.join(args_parser.path_logs_and_results, "output_code_carbon_offline.csv")
        elif self.name_calc == 'green_algorithms':
            self.automated = bool(args_parser.calculator_mode != "default")
            print('Calculator mode: ', args_parser.calculator_mode, ' automated=', self.automated)
            if self.automated:
                self.parallel = bool(args_parser.calculator_mode == "automated_parallel")
                print('Automated and parallel mode: ', self.parallel)

        self.eco2ai_output_file =  os.path.join(args_parser.path_logs_and_results, "output_eco2ai.csv")

        if self.name == "SQUAD-extracted" or self.name == "SQUAD-v1-1":
            # print('here')
            # print(tf.config.list_physical_devices('GPU'))
            # print(self.use_accelerator)
            # print(tf.config.list_physical_devices('GPU') != [])
            use_cuda = self.use_accelerator and (tf.config.list_physical_devices('GPU') != [])
            if use_cuda:
                os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
                self.device_name = "cuda"
            else:
                os.environ['CUDA_VISIBLE_DEVICES'] = ""
                self.device_name = "cpu"
        else:
            use_cuda = self.use_accelerator and torch.cuda.is_available()
            use_mps = self.use_accelerator and torch.backends.mps.is_available()
            if use_cuda:
                self.device = torch.device("cuda")
            elif use_mps:
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
            self.device_name = self.device.type
            print("Device is: ", self.device)

        if use_cuda: # rtx 2080 super info
            TDP = 250 # Watts (=J per sec)
            peak_flops = 11.15 # theorical TFLOPS (teraFLOPS, 10**12 FLOPS)
            self.perf_per_watt = peak_flops/TDP*10**12 # FLOPs per J
        else: # intel core i9-9900
            TDP = 95 # Watts
            peak_flops = 460.8 # theorical GFLOPS (gigaFLOPS, 10**9 FLOPS)
            self.perf_per_watt = peak_flops/TDP*10**9



def prepare_calculator(exp):
    #     Preparing the calculators     #

    tracker = None

    if exp.name_calc == 'code_carbon':
        output_file = exp.cc_output_file
        tracking_mode = 'machine'
        # tracking_mode = 'process'
        if exp.online == True:
            measure_power_secs = 2
            tracker = EmissionsTracker(
                output_file = output_file, 
                measure_power_secs = measure_power_secs, 
                tracking_mode = tracking_mode)
        else:
            country_iso_code = 'FRA'
            tracker = OfflineEmissionsTracker(
                output_file = output_file, 
                country_iso_code = country_iso_code)
            
    elif exp.name_calc == 'carbon_tracker':
        log_dir = exp.ct_log_dir
        # # Delete the previous logs:
        # for f in os.listdir(log_dir):
        #     if f != "readme.txt":
        #         os.remove(os.path.join(log_dir, f))
        update_interval = 2    # interval in seconds between power usage measurements are taken
        monitor_epochs = -1    # number of epochs that we want to monitor (-1 means all epochs)
        decimal_precision = 10 # desired decimal precision of reported values.
        if exp.measure == True:
            epochs_before_pred = 0 # number of epochs to use for average used for making the prediction
        else:
            epochs_before_pred = 1
        if exp.ml == 'training':
            # carbon_tracker_epochs = exp.epochs
            carbon_tracker_epochs = int(exp.epochs)
        elif exp.ml == 'inference':
            carbon_tracker_epochs = 1 # fake single epoch
        tracker = CarbonTracker(epochs=carbon_tracker_epochs, 
            update_interval = update_interval, 
            log_dir = log_dir, 
            monitor_epochs = monitor_epochs, 
            epochs_before_pred = epochs_before_pred, 
            decimal_precision = decimal_precision)

    elif exp.name_calc == 'eco2ai':
        output_file = exp.eco2ai_output_file
        alpha_2_code="FR"
        tracker = eco2ai.Tracker(file_name=output_file, 
            alpha_2_code=alpha_2_code, 
            pue=1, 
            measure_period=2)
    
    elif exp.name_calc == 'energy_scopium':
        # os.system("${ENERGY_SCOPE_SRC_DIR}/energy_scope_record.sh start")
        os.system("${ENERGYSCOPIUM_SRC_DIR}/energyscopium_record.sh start")
    
    elif exp.name_calc == 'green_algorithms':
        if exp.automated and exp.parallel:
            wait_for_UTIL()

    elif exp.name_calc == 'tapo':
        wait_for_TAPO()

    elif exp.name_calc == 'mygmlc' and exp.automated:
        tracker = MyGMLC(output_file = exp.mygmlc_output_file, 
                         country='France')

    return tracker
    

def start_calculators(exp, tracker):

    # timestamp - task start 
    tag = exp.name_calc + ':' + exp.mode_calc + ' task start'
    new_timestamp(exp.main_folder, exp.iteration, tag)

    if (exp.name_calc == 'code_carbon') or (exp.name_calc == 'eco2ai'):
        tracker.start()       
    elif (exp.name_calc == 'carbon_tracker') and (exp.ml == 'inference'):
        tracker.epoch_start()
    elif exp.name_calc == 'energy_scopium':
        # os.system("${ENERGY_SCOPE_SRC_DIR}/energy_scope_record.sh tags start tag_" + exp.ml)
        os.system("${ENERGYSCOPIUM_SRC_DIR}/energyscopium_record.sh tags start tag_" + exp.ml)
    elif exp.name_calc == 'mygmlc' and exp.automated:
        tracker.start()


def stop_calculators(exp, tracker):

    # timestamp - task stop
    tag = exp.name_calc + ':' + exp.mode_calc + ' task stop'
    new_timestamp(exp.main_folder, exp.iteration, tag)

    if (exp.name_calc == 'carbon_tracker') and (exp.ml == 'inference'):
        tracker.epoch_end() 
    elif (exp.name_calc == 'code_carbon') or (exp.name_calc == 'eco2ai') or (exp.name_calc == 'carbon_tracker'):
        tracker.stop()
    elif exp.name_calc == 'energy_scopium':
        # os.system("${ENERGY_SCOPE_SRC_DIR}/energy_scope_record.sh tags stop tag_" + exp.ml)
        os.system("${ENERGYSCOPIUM_SRC_DIR}/energyscopium_record.sh tags stop tag_" + exp.ml)
        os.system("${ENERGYSCOPIUM_SRC_DIR}/energyscopium_record.sh stop")
        # os.system("${ENERGYSCOPIUM_SRC_DIR}/energyscopium_record.sh send")
    elif exp.name_calc == 'mygmlc' and exp.automated:
        tracker.stop()

def FLOPs_inference_to_training(exp, nb_FLOPs_forward, nb_examples):

    # nb of times we do: forward pass (inference) + loss computation + backward pass
    nb_iterations = (nb_examples / exp.train_batch_size) * exp.epochs

    # coeff 3 comes from a blog:            
    factor = nb_iterations * 3 
    nb_FLOPs = nb_FLOPs_forward * factor

    print('Total nb FLOPs of training ', nb_FLOPs)
    return(nb_FLOPs)

def FLOPs_to_energy(exp, nb_FLOPs):
    Ec_J = nb_FLOPs/ exp.perf_per_watt
    Ec_kWh = Ec_J/(3.6 * 10**6) # (3.6 * 10**6) = nb of J per kWh
    print('Energy consumed (kWh): ', Ec_kWh)
    return(Ec_kWh)

def flops_method_pytorch(exp, data_loader, model):

    if exp.name_calc == 'flops':
        if exp.name == 'cifar10':
            input_channels = 3
            input_x = 32
            input_y = 32
        elif exp.name == 'mnist':
            input_channels = 1
            input_x = 28
            input_y = 28
        elif (exp.name == 'CUB_200_2011') or (exp.name == 'image_net'):
            input_channels = 3
            input_x = 224
            input_y = 224
        
        import torch
        from thop import profile
        nb_examples = len(data_loader)*exp.train_batch_size

        if (exp.ml == "training"):  
            input = torch.randn(exp.train_batch_size, input_channels, input_x, input_y)
            # the output of profile is in MACs (not gigaMACs or other)            
            MACs, params = profile(model,inputs=(input, ))
            nb_FLOPs_inference = MACs * 2
            print('Total nb of FLOPs inference: ', nb_FLOPs_inference)
            nb_FLOPs = FLOPs_inference_to_training(exp, nb_FLOPs_inference, nb_examples)
        
        elif (exp.ml == "inference"):  
            input = torch.randn(exp.test_batch_size, input_channels, input_x, input_y)
            MACs, params = profile(model,inputs=(input, ))
            nb_FLOPs = MACs* 2 
            print('Total nb of FLOPs inference: ', nb_FLOPs)

        Ec_kWh = FLOPs_to_energy(exp, nb_FLOPs)
        return Ec_kWh

def flops_method_tensorflow(exp, nb_examples, graph_folder):

    if exp.name_calc == 'flops':
        import tensorflow as tf
        import os
        from google.protobuf import text_format
        from tensorflow.python.platform import gfile

        def pbtxt_to_graphdef(filename):
            print("\n[CONVERT] Converting from .pbtxt to .pb: '{}'\n".format(filename))
            with open(filename, 'r') as f:
                graph_def = tf.compat.v1.GraphDef()
                file_content = f.read()
                text_format.Merge(file_content, graph_def)
                tf.import_graph_def(graph_def, name='')
                in_dir = os.path.dirname(filename)
                out_filename = os.path.splitext(os.path.basename(filename))[0] + ".pb"
                tf.compat.v1.train.write_graph(graph_def, in_dir, out_filename, as_text=False)
            print("\n[CONVERT] Wrote file to: '{}'\n".format(os.path.join(in_dir, out_filename)))

        file_name = os.path.join(graph_folder, 'graph.pbtxt')
        pbtxt_to_graphdef(file_name)

        def load_pb(pb):
            with tf.compat.v1.gfile.GFile(pb, "rb") as f:
                graph_def = tf.compat.v1.GraphDef()
                graph_def.ParseFromString(f.read())
            with tf.Graph().as_default() as graph:
                tf.import_graph_def(graph_def, name='')
                return graph

        g = load_pb(os.path.join(graph_folder, 'graph.pb'))

        with g.as_default():
            flops = tf.compat.v1.profiler.profile(g, options = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation())
            print('Total nb FLOPs of forward pass', flops.total_float_ops)

        nb_FLOPs = flops.total_float_ops
        if exp.ml == "training":
            nb_FLOPs = FLOPs_inference_to_training(exp, nb_FLOPs, nb_examples)

        Ec_kWh = FLOPs_to_energy(exp, nb_FLOPs)
        return Ec_kWh