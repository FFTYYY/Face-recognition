import os , sys

from config import C , logger

import fastNLP
from fastNLP import Vocabulary , DataSet , Instance , Tester
from fastNLP import Trainer , AccuracyMetric , CrossEntropyLoss
import torch.nn as nn
import torch as tc
from torch.optim import Adadelta
import numpy as np
import random
import pickle
from tqdm import tqdm
import torchvision
import torchvision.transforms as transforms

from model.resnet import Model as ResNet

from dataloader_lfw import load_data as load_data_lfw
from optim import *
from utils.confirm_tensor import tensor_feature

import pdb



#---------------------------------------------------------------------------------------------------
#Get data
data_loaders = {
	"lfw" 			: load_data_lfw,
}

train_data , test_data = data_loaders[C.data](dataset_location = C.data_path)

trainloader = tc.utils.data.DataLoader(train_data , batch_size = C.batch_size , shuffle = True , num_workers = 2)
testloader  = tc.utils.data.DataLoader(test_data  , batch_size = 20 		  , shuffle = False, num_workers = 2)

logger.log ("Data load done.")

#---------------------------------------------------------------------------------------------------
#Get model

models = {
	"resnet" 		: ResNet,
}
model = models[C.model]
net = model(num_class = 2 , input_size = [C.fmap_size[0] , C.fmap_size[0]] ,
	**{x : C.__dict__[x] for x in model.choose_kwargs()}
)

logger.log ("Creat network done.")

#---------------------------------------------------------------------------------------------------
#Valid

def valid(net , valid_data , epoch_num = 0):

	net = net.eval()

	tota_hit = 0
	good_hit = 0

	pbar = tqdm(testloader , ncols = 70)	
	pbar.set_description_str("(Epoch %d) Testing " % (epoch_num+1))

	for (xs, goldens) in pbar:

		xs = xs.cuda()
		goldens = goldens.cuda()
		ys = net(xs)["pred"]

		got = tc.max(ys , -1)[1]
		good_hit += int((goldens == got).long().sum())
		tota_hit += len(goldens)
		pbar.set_postfix_str("Valid Acc : %d/%d = %.4f%%" % (good_hit , tota_hit , 100 * good_hit / tota_hit))

	return good_hit , tota_hit

#---------------------------------------------------------------------------------------------------
#Optimizer control



#---------------------------------------------------------------------------------------------------
#Train
logger.log("Training start.")
logger.log("--------------------------------------------------------------------")

#variables about model saving
model_save_path = os.path.join(C.model_path , C.model_save)
best_acc = 0.
best_epoch = 0.

#optimizer
optims = {
	"warm_adam" : lambda : WarmAdam(params = net.parameters() , d_model = 256 , n_warmup_steps = 4000) ,
	"step_adam" : lambda : StepAdam(params = net.parameters() , lr = C.lr) ,
	"mysgd"  : lambda : MySGD (params = net.parameters() , lr = C.lr) , 
	"adam" 	 : lambda : tc.optim.Adam(params = net.parameters() , lr = C.lr) , 
	"sgd" 	 : lambda : tc.optim.SGD (params = net.parameters() , lr = C.lr) , 
}

optim = optims[C.optim]()


#loss function
loss_func = nn.CrossEntropyLoss()

net = nn.DataParallel(net , C.gpus)
#net = net.cuda()

tot_step = 0
for epoch_num in range(C.n_epochs):

	net = net.train()
	tota_hit = 0
	good_hit = 0
	pbar = tqdm(trainloader , ncols = 70)
	pbar.set_description_str("(Epoch %d) Training" % (epoch_num+1))
	for (xs, goldens) in pbar:

		xs = xs.cuda()
		goldens = goldens.cuda()
		ys = net(xs)["pred"]

		loss = loss_func(ys , goldens)
		optim.zero_grad()
		loss.backward()
		optim.step()

		got = tc.max(ys , -1)[1]
		good_hit += int((goldens == got).long().sum())
		tota_hit += len(goldens)
		pbar.set_postfix_str("Train Acc : %d/%d = %.4f%%" % (good_hit , tota_hit , 100 * good_hit / tota_hit))
		tot_step += 1

	valid_res = valid(net , test_data , epoch_num = epoch_num)

	logger.log("Epoch %d ended." % (epoch_num + 1))
	logger.log("Train Acc : %d/%d = %.4f%%" % (good_hit , tota_hit , 100 * good_hit / tota_hit))
	logger.log("Test  Acc : %d/%d = %.4f%%" % (valid_res[0] , valid_res[1] , 100 * valid_res[0] / valid_res[1]))
	logger.log("now total step = %d" % (tot_step))

	valid_acc = valid_res[0] / valid_res[1]
	if valid_acc > best_acc:
		best_acc = valid_acc
		best_epoch = epoch_num

	net = net.cpu()
	with open(model_save_path , "wb") as fil:
		pickle.dump(net.module , fil)
	net = net.cuda()
	logger.log("Model saved.")

	logger.log("--------------------------------------------------------------------")

logger.log("Best  Accurancy: %.4f%% in epoch %d" % (best_acc  , best_epoch))
logger.log("Final Accurancy: %.4f%% in epoch %d" % (valid_acc , C.n_epochs))