import torch as tc
from torch import nn
import torch.nn.functional as F
import fastNLP
import math
import pdb
from .sublayers import ResNetLayer_1 as Layer

class Model(nn.Module):
	def __init__(self, num_class = 2, input_size = [224,224] , 
		n = 9 , fmap_size = [224,112,56] , filter_num = [64,128,256] , drop_p = 0.0 , nores = False):

		super().__init__()

		assert input_size[0] == input_size[1] and input_size[1] == fmap_size[0]

		self.in_conv = nn.Conv2d(3 , filter_num[0] , kernel_size = 3 , padding = 1)
		#self.in_conv = nn.Conv2d(6 , filter_num[0] , kernel_size = 3 , padding = 1)
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

		#self.ln1 = nn.Linear(filter_num[-1] , 2 * filter_num[-1])
		self.ln1 = nn.Linear(2 * filter_num[-1] , 2 * filter_num[-1])
		self.lno = nn.Linear(2 * filter_num[-1] , num_class)


	def choose_kwargs():
		return ["n" , "fmap_size" , "filter_num" , "drop_p" , "nores"]

	def encode(self , s):
		s = self.drop_1(F.relu(self.in_bn(self.in_conv(s))))

		for layer in self.imm_layers:
			s = layer(s)

		bsz , d , len_1 , len_2 = s.size()

		s = s.view(bsz , d , len_1 * len_2)
		s = s.mean(dim = -1) #(bsz , filter_num)
		return s

	def forward(self , x):

		bs , _2 , _3 , n1 , n2 = x.size()
		assert _2 == 2 and _3 == 3

		#'''
		x1 , x2 = x[:,0] , x[:,1]
		x1 , x2 = self.encode(x1) , self.encode(x2)

		y = tc.cat([x1 , x2] , dim = -1)
		y = F.relu(self.ln1(y))
		y = self.lno(y)
		'''
		x = x.view(bs , 6 , n1 , n2)

		y = self.encode(x)

		y = F.relu(self.ln1(y))
		y = self.lno(y)
		'''

		return {
			"pred": y,
		}
