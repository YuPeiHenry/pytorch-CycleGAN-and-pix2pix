import torch
import torch.nn as nn

def simple_gradient(input, noise, epsilon):
	dx = 0.5 * (roll_x_plus1(input) - roll_x_minus1(input))
	dy = 0.5 * (roll_y_plus1(input) - roll_y_minus1(input))
	magnitude = torch.sqrt(dx * dx + dy * dy + (epsilon ** 2) / 10)

	randomX = noise
	randomY = torch.sqrt(1 - randomX * randomX)
	factor = torch.relu(epsilon - magnitude)
	
	final_dx = (dx + factor * randomX) / (magnitude + factor)
	final_dy = (dy + factor * randomY) / (magnitude + factor)
	
	# 4D Tensor, it is y, x. This is not a typo.
	return torch.cat((final_dy.unsqueeze(3), final_dx.unsqueeze(3)), 3)

def sample(input, offset, coord_grid, width):
	# coords are between [0, self.width - 1]. Normalize to [-1, 1]
	coords = coord_grid.repeat(input.size()[0], 1, 1, 1) + offset
	normalized = coords / (width - 1) * 2 - 1
	# For example, values: x: -1, y: -1 is the left-top pixel of the input
	# values: x: 1, y: 1 is the right-bottom pixel of the input
	return nn.functional.grid_sample((input - 1).unsqueeze(1), normalized, mode='bilinear', padding_mode='zeros').view(-1, width, width) + 1

def displace(a, delta):
	"""
	fns = {
		-1: lambda x: -x,
		0: lambda x: 1 - np.abs(x),
		1: lambda x: x,
	}"""
	delta_x, delta_y = delta[:, :, :, 0], delta[:, :, :, 1]
	delta_x = delta_x.unsqueeze(3)
	delta_y = delta_y.unsqueeze(3)
	# BatchSize x Height X Width x 3
	x_multipliers = torch.relu(torch.cat((-delta_x, 1 - torch.abs(delta_x), delta_x), 3))
	y_multipliers = torch.relu(torch.cat((-delta_y, 1 - torch.abs(delta_y), delta_y), 3))
	
	total = torch.zeros_like(a)
	result = torch.zeros_like(a)
	for dx in range(-1, 2):
		for dy in range(-1, 2):
			total = total + x_multipliers[:, :, :, 1 + dx] * y_multipliers[:, :, :, 1 + dy]
	for dx in range(-1, 2):
		for dy in range(-1, 2):
			temp = x_multipliers[:, :, :, 1 + dx] * y_multipliers[:, :, :, 1 + dy] * a / total
			result = result + roll_2d(temp, dx, dy)

	return result

def roll_2d(xy, dx, dy):
	if dx < 0:
		xy = roll_x_minus1(xy)
	elif dx > 0:
		xy = roll_x_plus1(xy)
	if dy < 0:
		xy = roll_y_minus1(xy)
	elif dy > 0:
		xy = roll_y_plus1(xy)
	return xy

def roll_y_plus1(y):
	return torch.cat((torch.zeros_like(y[:, 0]).unsqueeze(1), y[:, :-1]), 1)

def roll_y_minus1(y):
	return torch.cat((y[:, 1:], torch.zeros_like(y[:, 0]).unsqueeze(1)), 1)

def roll_x_plus1(x):
	return torch.cat((torch.zeros_like(x[:, :, 0]).unsqueeze(2), x[:, :, :-1]), 2)

def roll_x_minus1(x):
	return torch.cat((x[:, :, 1:], torch.zeros_like(x[:, :, 0]).unsqueeze(2)), 2)
