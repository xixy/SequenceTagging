#coding=utf-8
import os
import tensorflow as tf

class base_model(object):
	"""docstring for BaseModel"""
	def __init__(self, config, logger):
		self.config = config
		self.sess = None
		self.saver = None
		self.logger = logger

	def reinitialize_weights(self, scope_name):
		'''
		对某一scope的变量进行重新初始化
		'''
		variables = tf.contrib.framework.get_variables(scope_name)
		init = tf.variables_initializer(variables)
		self.sess.run(init)

	def add_train_op(self, lr_method, lr, loss, clip = -1):
		'''
		定义self.train_op来进行训练操作
		Args:
			lr_method: adam/adagrad/sgd/rmsprop
			lr: learning rate
			loss: 损失函数
			clip: 梯度的clipping值，如果<0，那么就不clipping
		'''
		_lr_m = lr_method.lower()
		with tf.variable_scope("train_step"):
			if _lr_m == 'adam':
				optimizer = tf.train.AdamOptimizer(lr)
			elif _lr_m == 'adagrad':
				optimizer = tf.train.AdagradOptimizer(lr)
			elif _lr_m == 'sgd':
				optimizer = tf.train.GradientDescentOptimizer(lr)
			elif _lr_m == 'rmsprop':
				optimizer = tf.train.RMSPropOptimizer(lr)
			else:
				raise NotImplementedError("Unknown method {}".format(_lr_m))

		# 进行clip
		if clip > 0:
			grads, vs = zip(*optimizer.compute_gradients(loss))
			grads, gnorm  = tf.clip_by_global_norm(grads, clip)
			self.train_op = optimizer.apply_gradients(zip(grads, vs))
		else:
			self.train_op = optimizer.minimize(loss)

	def initialize_session(self):
		'''
		定义self.sess并且初始化变量和self.saver
		'''
		self.logger.info("Initializing tf session")
		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())
		self.saver = tf.train.Saver()

	def restore_session(self, dir_model = None):
		'''
		reload weights into self.session
		Args:
			dir_model: 模型路径
		'''
		if dir_model is None:
			# 从config中读取模型路径
			dir_model = self.config['dir_model']

		self.logger.info("Reloading the latest trained model...")
		self.saver.restore(self.sess, dir_model)

	def save_session(self):
		'''
		存储session
		'''
		if not os.path.exists(self.config['dir_model']):
			os.makedirs(self.config['dir_model'])
		self.saver.save(self.sess, self.config['dir_model'])

	def close_session(self):
		'''
		close Session
		'''
		self.sessin.close()

	def add_summary(self):
		'''
		为tensorboard定义变量, 输出文件为dir_output

		'''
		self.merged = tf.summary.merge_all()
		self.file_writer = tf.summary.FileWriter(self.config["dir_output"],
			self.sess.graph)

	def train(self, train, dev):
		'''
		进行训练，采用了early stopping和学习率指数递减
		Args:
			train: dataset yields tuple of (sentence, tags)
			dev: dataset
		'''
		best_score = 0
		num_of_epoch_no_imprv = 0 # for early stopping，用来记录几个epoch没有提高了
		self.add_summary() # tensorboard
		for epoch in range(self.config['epochs']):
			self.logger.info('Epoch {:} out of {:}'.format(epoch + 1, self.config['epochs']))
			
			# 运行一个epoch的训练工作，并返回在dev数据集上的测试f1
			score = self.run_epoch(train, dev, epoch)
			# 进行learning rate decay
			self.config['learning_rate'] *= self.config['learning_rate_decay']

			# 进行early stopping并且保存最好的参数
			# 如果效果更好了
			if score >= best_score:
				# 清零
				num_of_epoch_no_imprv = 0
				# 记录当前的参数
				self.save_session()
				# 更新best score
				best_score = score
				self.logger.info("- new best score! ")
			# 如果效果没有更好
			else:
				num_of_epoch_no_imprv += 1
				# 如果已经好多轮没有效果更好了，就需呀stop
				if num_of_epoch_no_imprv >= self.config['num_of_epoch_no_imprv']:
					self.logger.info("- early stopping {} epochs without "\
						"improvement".format(nepoch_no_imprv))
					break


	def evaluate(self, test):
		'''
		在测试集上对模型进行测试
		Args:
			test: datset from test.txt
		'''
		self.logger.info("Testing model over test set")
		# 跑测试
		metrics = self.run_evaluate(test)
		msg = " - ".join(["{} {:04.2f}".format(k, v) for k, v in metrics.items()])
		self.logger.info(msg)







