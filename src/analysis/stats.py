import numpy as np


def cumsum_sample(data):
	data_sorted = np.sort(data)

	# calculate the proportional values of samples
	# all of the above are unbiased estimations of the CDF
	# p = 1. * np.arange(len(data)) / (len(data) - 1)
	# p = (arange(len(x))+0.5)/len(x)
	p = np.linspace(0, 1, len(data), endpoint=False)
	return data_sorted, p
