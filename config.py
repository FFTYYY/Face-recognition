import argparse
import os
from utils.logger import Logger
from pprint import pformat
from fastNLP.core._logger import logger as inner_logger
import random
import torch as tc
import numpy as np

_par = argparse.ArgumentParser()

#---------------------------------------------------------------------------------------------------

#dataloader
_par.add_argument("--data" 			, type = str , default = "lfw" , choices = ["lfw"])
_par.add_argument("--data_path" 	, type = str , default = "datas/")
_par.add_argument("--force_reprocess" 	, action = "store_true", default = False)

#model universal
_par.add_argument("--model" 		, type = str , default = "resnet" , choices = [
	"resnet" , 
])
_par.add_argument("--drop_p" 		, type = float , default = 0.0)

#model resnet
_par.add_argument("--fmap_size" 	, type = str , default = "32,16,8")
_par.add_argument("--filter_num" 	, type = str , default = "16,32,64")
_par.add_argument("--nores" 		, action = "store_true" , default = False)

#train & test
_par.add_argument("--batch_size" 	, type = int , default = 64)
_par.add_argument("--n_epochs" 		, type = int , default = 32)
_par.add_argument("--lr" 			, type = float , default = 1e-3)
_par.add_argument("--gpus" 			, type = str , default = "0")
_par.add_argument("--valid_size" 	, type = int , default = 1000)
_par.add_argument("--init_steps" 	, type = int , default = 0)
_par.add_argument("--step_size" 	, type = int , default = 1)
_par.add_argument("--valid_data" 	, type = str , default = "test")
_par.add_argument("--optim" 		, type = str , default = "myadam" , choices = ["myadam" , "adam" , "sgd" , "mysgd"])


#solely test
_par.add_argument("--test_mode" 	, action = "store_true" , default = False)
_par.add_argument("--model_path" 	, type = str , default = "trained_models/")
_par.add_argument("--model_save" 	, type = str , default = "trained.pkl")

#others
_par.add_argument("--seed" 			, type = int , default = 2333)
_par.add_argument("--log_file" 		, type = str , default = "log.txt")
_par.add_argument("--no_log" 		, action = "store_true" , default = False)
_par.add_argument("--file" 			, type = str , default = "")

#---------------------------------------------------------------------------------------------------

C = _par.parse_args()

C.data_path = os.path.join(C.data_path , C.data)

def listize(name):
	C.__dict__[name] = [int(x) for x in filter(lambda x:x , C.__dict__[name].strip().split(","))]

listize("gpus")
listize("fmap_size")
listize("filter_num")

if C.test_mode:
	C.log_file += ".test"

logger = Logger(inner_logger , C.log_file)
logger.log = logger.log_print_w_time
if C.no_log:
	logger.log = logger.nolog

C.input_size = C.fmap_size[0]

logger.log ("------------------------------------------------------")
logger.log (pformat(C.__dict__))
logger.log ("------------------------------------------------------")

#Initialize

if C.seed > 0:
	random.seed(C.seed)
	tc.manual_seed(C.seed)
	np.random.seed(C.seed)
	tc.cuda.manual_seed_all(C.seed)
	tc.backends.cudnn.deterministic = True
	tc.backends.cudnn.benchmark = False

	logger.log ("Seed set. %d" % (C.seed))

tc.cuda.set_device(C.gpus[0])