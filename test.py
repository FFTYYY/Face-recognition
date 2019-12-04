import os , sys

import fastNLP
from fastNLP import Vocabulary , DataSet , Instance , Tester
from fastNLP import Trainer , AccuracyMetric , CrossEntropyLoss , Adam
import torch.nn as nn
import torch as tc
from torch.optim import Adadelta
import numpy as np
import random
import pickle
from tqdm import tqdm
import torchvision.transforms as transforms


from model.multi_dim_transformer import Model as MD_Transformer
from model.resnet import Model as ResNet
from model.third_party import ResNet_56_3p
from config import C , logger
from dataloader_cifar10 import n_crop_test as n_crop_test_cifar_10

import pdb

#---------------------------------------------------------------------------------------------------
#Get data
data_loaders = {
	"cifar-10" 			: n_crop_test_cifar_10,
}

test_data = data_loaders[C.data](dataset_location = C.data_path , n_crop = C.n_crop)

logger.log ("Data load done.")
logger.log ("test size = %d" % (len(test_data)))
#---------------------------------------------------------------------------------------------------
#Get model

with open(os.path.join(C.model_path , C.model_save) , "rb") as fil:
	net = pickle.load( fil ).cuda( C.gpus[0] )

logger.log ("Load network done.")
#---------------------------------------------------------------------------------------------------
#fastNLP Test

if C.n_crop <= 1:
	fastNLP_data = DataSet()
	for s  , lab in test_data:
		fastNLP_data.append ( Instance (s = s[0] , label = lab) )
	fastNLP_data.set_input("s")
	fastNLP_data.set_target("label")

	tester = Tester(
		data 	= fastNLP_data , 
		model 	= net , 
		metrics = AccuracyMetric(pred = "pred" , target = "label") ,
		device 	= C.gpus , 
	)
	test_result = tester.test()
	logger.log("fastNLP test: {0}".format(test_result))


#---------------------------------------------------------------------------------------------------
#Test

net = net.eval()
good_hit = 0
tot_hit = 0
pbar = tqdm(test_data , ncols = 70)
for data in pbar:
	ss , gold = data

	with tc.no_grad():
		s = tc.cat( [s.unsqueeze(0) for s in ss] , dim = 0)
		y = net( s.cuda( C.gpus[0] ) )["pred"]
		y = tc.softmax(y , dim = -1).mean(dim = 0)

	y = int(tc.max(y , -1)[1])
	if y == gold:
		good_hit += 1
	tot_hit += 1
	pbar.set_postfix_str("Test Accurancy : %d/%d = %.2f%%" % (good_hit , tot_hit , 100 * good_hit / tot_hit))
logger.log ("Test Accurancy : %d/%d = %.2f%%" % (good_hit , len(test_data) , 100 * good_hit / len(test_data)))

#---------------------------------------------------------------------------------------------------
