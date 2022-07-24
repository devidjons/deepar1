import pdb

from torch import mean, log, exp


def gausian_prob(input, target):
	# pdb.set_trace()
	
	mu = input[0]
	sigma = input[1]
	# return mean(log(sigma) + (1.0/2.0*(target[skip:] - mu)**2)/sigma)
	return mean((target - mu) ** 2 + 0 * sigma)
