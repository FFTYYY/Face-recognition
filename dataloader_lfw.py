import torch as tc
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import time
import os
import pdb
import sys
import pickle
import random
import PIL
import copy

dataset_loc = ""

transform_train = transforms.Compose([
	transforms.Resize(32),
	transforms.RandomCrop(32 , padding = 4),
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor(),
	transforms.Normalize(mean = [0.491 , 0.482 , 0.447] , std = [0.247 , 0.243 , 0.262]),
])

transform_test  = transforms.Compose([	
	transforms.Resize(32),
	transforms.ToTensor(),
	transforms.Normalize(mean = [0.491 , 0.482 , 0.447] , std = [0.247 , 0.243 , 0.262]),
])


def jpath(path):
	return os.path.join(dataset_loc , path)

def load_a_image(name , idx , trans):
	path = "./%s/%s_%04d.jpg" % (name , name , int(idx))
	_img = PIL.Image.open(jpath(path))
	img = copy.deepcopy(_img)
	_img.close()
	return trans(img)

def load_a_set(fil , trans):

	datas = []

	num_posi = int(fil.readline().strip())
	for i in range(num_posi):
		person_name , id1 , id2 = fil.readline().strip().split("\t")
		datas.append(
			[
				load_a_image(person_name , id1 , trans) , 
				load_a_image(person_name , id2 , trans) , 
				1 , 
			]
		)
	while fil:
		got = fil.readline().strip().split("\t")
		if len(got) < 4:
			break
		person_name_1 , id1 , person_name_2 , id2 = got
		datas.append(
			[
				load_a_image(person_name_1 , id1 , trans) , 
				load_a_image(person_name_2 , id2 , trans) , 
				0 , 
			]
		)
	return datas


def load_data(dataset_location = "./datas"):
	global dataset_loc
	dataset_loc = dataset_location

	with open(jpath("pairsDevTrain.txt") , "r" , encoding = "utf-8") as fil:
		train_data = load_a_set(fil , transform_train)
	with open(jpath("pairsDevTest.txt") , "r" , encoding = "utf-8") as fil:
		test_data = load_a_set(fil , transform_test)

	random.shuffle(train_data)
	random.shuffle(test_data)

	return train_data , test_data



if __name__ == "__main__":
	from config import C
	dat = load_data(dataset_location = C.data_path)


	pdb.set_trace()