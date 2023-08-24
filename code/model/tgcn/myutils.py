import torch
class BertLayerNorm(torch.nn.Module):
	"""This class is LayerNorm model for Bert
	"""
	
	def __init__(self, hidden_size, eps=1e-12):
		"""This function sets `BertLayerNorm` parameters

		Arguments:
			hidden_size {int} -- input size

		Keyword Arguments:
			eps {float} -- epsilon (default: {1e-12})
		"""
		
		super().__init__()
		self.weight = torch.nn.Parameter(torch.ones(hidden_size))  # , requires_grad=False)
		self.bias = torch.nn.Parameter(torch.zeros(hidden_size))  # , requires_grad=False)
		self.variance_epsilon = eps
	
	def forward(self, x):
		"""This function propagates forwardly

		Arguments:
			x {tensor} -- input tesor

		Returns:
			tensor -- LayerNorm outputs
		"""
		
		u = x.mean(-1, keepdim=True)
		s = (x - u).pow(2).mean(-1, keepdim=True)
		x = (x - u) / torch.sqrt(s + self.variance_epsilon)
		return self.weight * x + self.bias


import math


def gelu(x):
	"""Implementation of the gelu activation function.
		For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
		0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
		Also see https://arxiv.org/abs/1606.08415
		refer to: https://github.com/huggingface/pytorch-transformers/blob/master/pytorch_transformers/modeling_bert.py
	"""
	
	return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class BertLinear(torch.nn.Module):
	"""This class is Linear model for Bert
	"""
	
	def __init__(self, input_size, output_size, activation=gelu, dropout=0.0):
		"""This function sets `BertLinear` model parameters

		Arguments:
			input_size {int} -- input size
			output_size {int} -- output size

		Keyword Arguments:
			activation {function} -- activation function (default: {gelu})
			dropout {float} -- dropout rate (default: {0.0})
		"""
		super().__init__()
		self.input_size = input_size
		self.output_size = output_size
		self.linear = torch.nn.Linear(input_size, output_size)
		self.linear.weight.data.normal_(mean=0.0, std=0.02)
		self.linear.bias.data.zero_()
		self.activation = activation
		self.layer_norm = BertLayerNorm(self.output_size)
		if dropout > 0:
			self.dropout = torch.nn.Dropout(p=dropout)
		else:
			self.dropout = lambda x: x
	
	def get_input_dims(self):
		return self.input_size
	
	def get_output_dims(self):
		return self.output_size
	
	def forward(self, x):
		output = self.activation(self.linear(x))
		return self.dropout(self.layer_norm(output))
