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
		assert vi.shape[-1] == vq.shape[-1], "Dimentions of Vi and Vq don't match during Attention"
		a = self.w1(vi)
		# unsqueeze for adding vector to matrix.
		# read about broadcasting: https://pytorch.org/docs/stable/notes/broadcasting.html
		b = torch.unsqueeze(self.w2(vq), -2)
		# h.shape: (m, hidden_dim)
		h = self.tanh(a + b)
		# pi.shape: (m, 1)
		pi = F.softmax(self.wp(h), dim=1)
		return pi
	
	
class AttentionLayer(nn.Module):
	def __init__(self, hidden_dim, img_dim, ques_dim):
		super(AttentionLayer, self).__init__()
		self.attention = Attention(hidden_dim, img_dim, ques_dim)

	def forward(self, vi, u):
		"""Implements Stacked Attention Layer. Question feature 'u' is repeatedly passed through this layer,
		it calculates the attention over the image features, and merges the features into itself.

		Args:
			vi (m, d): Visual features of image, where m = (width * height) of feature map
			u (d):     Features of the question

		Returns:
			u (d): Features of question after applying the attention over the image
		"""
		pi = self.attention(vi, u)
		assert pi.shape[1] == vi.shape[1], f"pi ({pi.shape}) and vi ({vi.shape}) don't have same shape in AttentionLayer"
		u = torch.sum(pi * vi) + torch.unsqueeze(u, dim=-2)
		return torch.squeeze(u, dim=-2)
	
	
if __name__ == '__main__':
	m, d, hidden_dim = (14 * 14), 120, 53
	bs = 2

	vi = torch.rand((bs, m, d))
	u = torch.rand((bs, d))

	model = Attention(hidden_dim, d, d)
	out = model(vi, u)

	assert out.shape == (bs, m, 1)
