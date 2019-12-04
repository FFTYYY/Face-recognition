import sys
from fastNLP import Optimizer
import torch as tc
import torch.nn as nn
import torch.optim as optim
import numpy as np
from config import logger

class MyAdam(optim.Optimizer):
	def __init__(self, params , d_model, n_warmup_steps , init_steps  , step_size):
		self.init_lr = np.power(d_model, -0.5)
		self._optimizer = optim.Adam(params = params , lr = self.init_lr , betas = (0.9,0.98))
		self.n_warmup_steps = n_warmup_steps
		self.now_step = init_steps
		self.step_size = step_size

	def step(self):
		self._update_learning_rate()
		self._optimizer.step()

	def zero_grad(self):
		self._optimizer.zero_grad()

	def _get_lr_scale(self):
		return np.min([
			np.power(self.now_step, -0.5),
			np.power(self.n_warmup_steps, -1.5) * self.now_step
		])

	def _update_learning_rate(self):

		self.now_step += self.step_size
		lr = self.init_lr * self._get_lr_scale()

		for param_group in self._optimizer.param_groups:
			param_group['lr'] = lr

class MySGD(optim.Optimizer):
	def __init__(self, params , lr):
		self._optimizer = optim.SGD(params = params , lr = lr , momentum = 0.9 , weight_decay = 1e-4)
		self.now_step = 0
		self.now_lr = lr
		self.barriers = [32000 , 48000]

	def step(self):
		self._update_learning_rate()
		self._optimizer.step()

	def zero_grad(self):
		self._optimizer.zero_grad()

	def _update_learning_rate(self):

		self.now_step += 1
		if self.now_step in self.barriers:
			self.now_lr *= 0.1
			logger.log("now lr changing.... new lr = %.4f" % (self.now_lr))
			for param_group in self._optimizer.param_groups:
				param_group['lr'] = self.now_lr

