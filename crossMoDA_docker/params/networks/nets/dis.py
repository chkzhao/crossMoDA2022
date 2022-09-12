import torch.nn as nn
import torch.nn.functional as F


class FCDiscriminator(nn.Module):

	def __init__(self, num_classes, ndf = 64):
		super(FCDiscriminator, self).__init__()

		self.num_classes = num_classes

		self.model = nn.Sequential(
			nn.Linear(num_classes,ndf),
			nn.ReLU(),
			nn.Linear(ndf,ndf),
			nn.ReLU(),
			nn.Linear(ndf,1)
		)
		#self.up_sample = nn.Upsample(scale_factor=32, mode='bilinear')
		self.sigmoid = nn.Sigmoid()


	def forward(self, x):
		x = x.permute(0,2,3,4,1)
		x = x.reshape(-1,self.num_classes)
		x = self.model(x)
		#x = self.up_sample(x)
		x = self.sigmoid(x)

		return x