import torch as tc
from torch import nn
import torch.nn.functional as F
import fastNLP
import math
import pdb

class ResNetLayer_1(nn.Module):
	def __init__(self , n_in_channels , n_out_channels , sub_sampling = False , drop_p = 0.0 , nores = False):
		'''
			subsampling : stride = 2 to decrease the feature map
		'''

		super().__init__()

		self.conv1 = nn.Conv2d(n_in_channels , n_in_channels , 
				kernel_size = 3 , padding = 1 , stride = 1)
		self.bn1 = nn.BatchNorm2d(n_in_channels)
		self.conv2 = nn.Conv2d(n_in_channels , n_out_channels , 
				kernel_size = 3 , padding = 1 , stride = 2 if sub_sampling else 1)
		self.bn2 = nn.BatchNorm2d(n_out_channels)

		self.proj = None
		if n_in_channels != n_out_channels:
			self.proj = nn.Conv2d(n_in_channels , n_out_channels , 
				kernel_size = 1 , padding = 0 , stride = 2 if sub_sampling else 1)

		self.dropout = nn.Dropout(drop_p)

		self.reset_parameters()

		self.nores = nores

	def reset_parameters(self):
		#nn.init.xavier_normal_(self.conv1.weight.data , gain = 1)
		#nn.init.xavier_normal_(self.conv2.weight.data , gain = 1)
		#if self.proj is not None:
		#	nn.init.xavier_normal_(self.proj.weight.data , gain = 1)
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				#n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				nn.init.xavier_normal_(m.weight.data)
				#m.weight.data.normal_(0, math.sqrt(2. / n))
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()


	def forward(self , x):
		old_x = x
		x = F.relu(self.bn1(self.conv1(x)))
		x = self.bn2(self.conv2(x))

		if self.proj is not None:
			old_x = self.proj(old_x)

		if not self.nores:
			x = x + old_x

		x = self.dropout(F.relu(x))

		return x