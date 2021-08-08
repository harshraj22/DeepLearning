import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
	def __init__(self, hidden_dim, img_dim, ques_dim):
		super(Attention, self).__init__()
		self.w1 = nn.Linear(img_dim, hidden_dim, bias=False)
		self.w2 = nn.Linear(ques_dim, hidden_dim)
		self.wp = nn.Linear(hidden_dim, 1)
		self.tanh = torch.tanh
	
	def forward(self, vi, vq):
		"""
		Summary: 
			d = img_dim = ques_dim
		Args:
			vi (m, d): Visual features of the image, where m = (width * height) of feature map
			vq (d):    Features of the question

		Returns:
			pi (m): Attention coeeficient (alpha) corresponding to each feature vector
		"""
		# h.shape: (hidden_dim, m)
		h = self.tanh(self.w1(vi) + self.w2(vq))
		# pi.shape: (m, 1)
		pi = F.softmax(self.wp(h))
		return pi
