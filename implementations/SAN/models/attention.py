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
		a = self.w1(vi)
		# unsqueeze for adding vector to matrix.
		# read about broadcasting: https://pytorch.org/docs/stable/notes/broadcasting.html
		b = torch.unsqueeze(self.w2(vq), -2)
		# h.shape: (m, hidden_dim)
		h = self.tanh(a + b)
		# pi.shape: (m, 1)
		pi = F.softmax(self.wp(h))
		return pi
	

if __name__ == '__main__':
	m, d, hidden_dim = (14 * 14), 120, 53
	bs = 2

	vi = torch.rand((bs, m, d))
	u = torch.rand((bs, d))

	model = Attention(hidden_dim, d, d)
	out = model(vi, u)

	assert out.shape == (bs, m, 1)
