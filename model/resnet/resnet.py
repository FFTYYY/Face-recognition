import torch as tc
from torch import nn
import torch.nn.functional as F
import fastNLP
import math
import pdb
from .sublayers import ResNetLayer_1 as Layer

class Model(nn.Module):
	def __init__(self, num_class = 10, input_size = [32,32] , 
		n = 9 , fmap_size = [32,16,8] , filter_num = [16,32,64] , drop_p = 0.0 , nores = False):

		super().__init__()

		assert input_size[0] == input_size[1] and input_size[1] == fmap_size[0]

		self.in_conv = nn.Conv2d(3 , filter_num[0] , kernel_size = 3 , padding = 1)
		self.in_bn = nn.BatchNorm2d(filter_num[0])
		self.drop_1 = nn.Dropout(drop_p)

		imm_layers = []

		for i in range(len(fmap_size)):
			for j in range(n):
				filter_size_changing = ( (j == n-1) and (i != len(fmap_size)-1) ) #此layer之后就要换新的filter size
				d_in  = filter_num[i]
				d_out = filter_num[i+1] if filter_size_changing else filter_num[i]

				imm_layers.append( Layer(d_in , d_out , filter_size_changing , drop_p = drop_p , nores = nores) )

		self.imm_layers = nn.ModuleList(imm_layers)

		self.out_ln = nn.Linear(filter_num[-1] , num_class)


	def choose_kwargs():
		return ["n" , "fmap_size" , "filter_num" , "drop_p" , "nores"]

	def forward(self , s):


		s = self.drop_1(F.relu(self.in_bn(self.in_conv(s))))

		for layer in self.imm_layers:
			s = layer(s)

		bsz , d , len_1 , len_2 = s.size()

		s = s.view(bsz , d , len_1 * len_2)
		s = s.mean(dim = -1) #(bsz , filter_num)

		s = self.out_ln(s)   #(bsz , num_class)

		return {
			"pred": s,
		}
