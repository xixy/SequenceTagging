#coding=utf-8
#配置文件

class Configure(object):
	"""用来保存配置和超参数的文件"""

	'''
	读取配置文件
	'''
	def __init__(self, conf_file = 'config.cfg'):
		self.confs = {}
		for line in open(conf_file):
			line = line.strip().split()
			if len(line) < 3:
				continue
			key, value, type = line
			self.confs[key] = eval(type +"('" + value + "')" )


	def __setitem__(self, key, value):
		self.confs[key] = value

	def __getitem__(self, key):
		return self.confs[key]

if __name__ == '__main__':
	configure = Configure('./config.cfg')
	print configure['train_data']
	print configure['test_data']