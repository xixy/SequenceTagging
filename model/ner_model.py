#coding=utf-8
from base_model import *
import numpy as np
class ner_model(base_model):
	"""NER model"""
	def __init__(self, config, logger, vocabs, embeddings = None):
		super(ner_model, self).__init__(config, logger)
		self.embeddings = embeddings
		self.vocabs = vocabs
		self.vocab_words = vocabs[0]
		self.vocab_chars = vocabs[1]
		self.vocab_tags = vocabs[2]

	def build(self):
		'''
		构建可计算图
		'''
		# 添加placeholders
		self.add_placeholders()
		# 添加向量化操作，得到每个词的向量表示
		self.add_word_embeddings_op()
		# 计算logits
		self.add_logits_op()
		# 计算概率
		self.add_pred_op()
		# 计算损失
		self.add_loss_op()

		# 定义训练操作
		self.add_train_op(self.config['optimizer'], self.lr, self.loss,
			self.config['clip'])
		# 创建session和logger
		self.initialize_session() 




	def add_placeholders(self):
		'''
		定义placeholder
		'''
		# 表示batch中每个句子的词的id表示
		# shape = (batch size, max length of sentence in batch)
		self.word_ids = tf.placeholder(tf.int32, shape = [None, None], name = "word_ids")

		# 表示batch中每个句子的长度
		# shape = (batch size,)
		self.sequence_lengths = tf.placeholder(tf.int32, shape = [None], name = "sequence_lengths")

		# 表示batch中每个句子的每个词的字母id表示
		# shape = (batch size, max length of sentence in batch, max length of word)
		self.char_ids = tf.placeholder(tf.int32, shape = [None, None, None],
			name = "char_ids")

		# 表示batch中每个句子的每个词的长度
		# shape = (batch size, max length of sentence in batch)
		self.word_lengths = tf.placeholder(tf.int32, shape = [None, None], name = "word_lengths")

		# 表示batch中每个句子的每个word的label
		# shape = (batch size, max length of sentence in batch)
		self.labels = tf.placeholder(tf.int32, shape = [None, None], name = "labels")

		# dropout
		self.dropout = tf.placeholder(tf.float32, shape = [], name = "dropout")
		# 学习率
		self.lr = tf.placeholder(tf.float32, shape = [], name = "lr")


	def add_word_embeddings_op(self):
		'''
		添加embedding操作，包括词向量和字向量
		如果self.embeddings不是None，那么词向量就采用pre-trained vectors，否则自行训练
		字向量是自行训练的
		'''
		with tf.variable_scope("words"):
			# 如果词向量是None
			if self.embeddings is None:
				self.logger.info("WARNING: randomly initializing word vectors")
				_word_embeddings = tf.get_variable(
					name = '_word_embeddings',
					dtype = tf.float32,
					shape = [len(self.vocab_words), self.config['word_embedding_dim']]
					)
			else:
				# 加载已有的词向量
				_word_embeddings = tf.Variable(
					self.embeddings,
					name = '_word_embeddings',
					dtype = tf.float32,
					trainable = self.config['training_embeddings']
					)
			# lookup来获取word_ids对应的embeddings
			# shape = (batch size, max_length_sentence, dim)
			word_embeddings = tf.nn.embedding_lookup(
				_word_embeddings,
				self.word_ids,
				name = 'word_embeddings'
				)
		with tf.variable_scope('chars'):
			if self.config['use_chars']:
				_char_embeddings = tf.get_variable(
					name = '_char_embeddings',
					dtype = tf.float32,
					shape = [len(self.vocab_chars), self.config['char_embedding_dim']]
					)
				# shape = (batch, max_length_sentence, max_length_word, dim of char embeddings)
				char_embeddings = tf.nn.embedding_lookup(
					_char_embeddings,
					self.char_ids,
					name = 'char_embeddings'
					)

				# 2. put the time dimension on axis=1 for dynamic_rnn
				s = tf.shape(char_embeddings)
				# shape = (batch * max_length_sentence, max_length_word, dim of char embeddings)
				char_embeddings = tf.reshape(char_embeddings,
					shape = [s[0]*s[1], s[-2], self.config['char_embedding_dim']]
					)
				
				# 表示batch中每个句子的每个词的长度
				# shape = (batch size * max_length_sentence,)
				word_lengths = tf.reshape(self.word_lengths, shape=[s[0]*s[1]])

				# 3 bi-lstm on chars
				cell_fw = tf.contrib.rnn.LSTMCell(self.config['hidden_size_char'],
					state_is_tuple = True
					)
				cell_bw = tf.contrib.rnn.LSTMCell(self.config['hidden_size_char'],
					state_is_tuple = True
					)
				_output = tf.nn.bidirectional_dynamic_rnn(
					cell_bw, # 前向RNN
					cell_bw, # 后向RNN
					char_embeddings, # 输入序列
					sequence_length = word_lengths, # 序列长度
					dtype = tf.float32
					)

				# 取出final_state h，而不是c
				# shape = (batch * max_length_sentence, hidden_size_char)
				_, ((_, output_fw), (_, output_bw)) = _output
				# 双向输出进行合并

				# shape = (batch * max_length_sentence, 2 * hidden_size_char)
				output = tf.concat([output_fw, output_bw], axis=-1)


				# reshpae到 shape = (batch size, max_length_sentence, 2 * char hidden size)
				output = tf.reshape(
					output,
					shape = [s[0], s[1], 2 * self.config['hidden_size_char']]
					)

				# 合并到word_embedding上
				# shape = (batch size, max_length_sentence, 2 * char hidden size + word vector dim)
				word_embeddings = tf.concat([word_embeddings, output], axis = -1)

		self.word_embeddings = tf.nn.dropout(word_embeddings, self.dropout)


	def add_logits_op(self):
		'''
		定义self.logits，句子中的每个词都对应一个得分向量，维度是tags的维度
		'''
		# 首先对句子进行LSTM
		with tf.variable_scope('bi-lstm'):
			cell_fw = tf.contrib.rnn.LSTMCell(
				self.config['hidden_size_lstm']
				)
			cell_bw = tf.contrib.rnn.LSTMCell(
				self.config['hidden_size_lstm']
				)
			output = tf.nn.bidirectional_dynamic_rnn(
				cell_fw,
				cell_bw,
				self.word_embeddings,
				sequence_length = self.sequence_lengths,
				dtype = tf.float32
				)
			# 取出output
			# shape = (batch size, max_length_sentence, hidden_size_lstm)
			(output_fw, output_bw), _ = output
			# shape = (batch size, max_length_sentence, 2 * hidden_size_lstm)
			output = tf.concat([output_fw, output_bw], axis = -1)
			# 进行dropout
			# shape = (batch size, max_length_sentence, 2 * hidden_size_lstm)
			output = tf.nn.dropout(output, self.dropout)

		# 然后用全联接网络计算概率
		with tf.variable_scope('proj'):
			W = tf.get_variable(
				name = 'w', 
				dtype = tf.float32,
				shape = [2 * self.config['hidden_size_lstm'], len(self.vocab_tags)]
				)
			b = tf.get_variable(
				name = 'b',
				dtype = tf.float32,
				shape = [len(self.vocab_tags)],
				initializer = tf.zeros_initializer()
				)
			# 取出max_length_sentence
			nsteps = tf.shape(output)[1]

			# shape = (batch size * max_length_sentence, 2 * hidden_size_lstm)
			output = tf.reshape(output, [-1, 2*self.config['hidden_size_lstm']])

			# shape = (batch size * max_length_sentence, vocab_tags_nums)
			pred = tf.matmul(output, W) + b

			# shape = (batch size, max_length_sentence, vocab_tags_nums)
			self.logits = tf.reshape(pred, [-1, nsteps, len(self.vocab_tags)])

	def add_pred_op(self):
		'''
		计算prediction，如果使用crf的话，需要
		'''
		# 取出概率最大的维度的idx
		# shape = (batch size, max_length_sentence)
		if not self.config['use_crf']:
			self.labels_pred = tf.cast(tf.argmax(self.logits, axis=-1), tf.int32)

	def add_loss_op(self):
		'''
		计算损失
		'''
		# 如果使用crf部分
		if self.config['use_crf']:
			log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(
				self.logits,
				self.labels,
				self.sequence_lengths
				)
			self.trans_params = trans_params
			self.loss = tf.reduce_mean(-log_likelihood)
		# 如果不计算crf部分
		else:
			# shape = (batch size, max_length_sentence)
			losses = tf.nn.sparse_softmax_cross_entrophy_with_logits(
				logits = self.logits,
				labels = self.labels
				)
			mask = tf.sequence_mask(self.sequence_lengths)
			losses = tf.boolean_mask(losses, mask)
			self.loss = tf.reduce_mean(losses)
		# for tensorboard
		tf.summary.scalar('loss', self.loss)


	def run_epoch(self, train, dev, epoch):
		'''
		运行一个epoch，包括在训练集上训练、dev集上测试，一个epoch中对train集合的所有数据进行训练
		'''
		batch_size = self.config['batch_size']
		nbatches = (len(train) + batch_size - 1) // batch_size


		for i, (words, labels) in enumerate(train.get_minibatch(batch_size)):

			# 构造feed_dict,主要是包括：
			# 1. word_ids, word_length
			# 2. char_ids, char_length
			# 3. learning rate
			# 4. dropout keep prob
			fd, _ = self.get_feed_dict(words, labels, 
				self.config['learning_rate'], self.config['dropout'])

			# 执行计算
			_, train_loss, summary = self.sess.run(
				[self.train_op, self.loss, self.merged],
				feed_dict = fd
				)

			# tensorboard
			if i % 10 == 0:
				self.file_writer.add_summary(summary, epoch * nbatches + i)

		# 在dev集上面进行测试
		metrics = self.run_evaluate(dev)

		msg = " - ".join(["{} {:04.2f}".format(k, v)
			for k, v in metrics.items()])
		self.logger.info(msg)

		# 返回f1
		return metrics["f1"]

	def pad_sequences(self, sequences, pad_token, nlevels = 1):
		'''
		对sequence进行填充
		Args:
			sequences: a generator of list or tuple
			pad_token: the token to pa with
			nlevels: padding的深度，如果是1，则表示对词进行填充，如果是2表示对字进行填充
		Return:
			a list of list where each sublist has same length
		'''
		print '--------pad-sequences--------'
		if nlevels == 1:
			# 找到sequences中的句子最大长度
			max_length_sentence = max(map(lambda x: len(x), sequences))
			# 然后直接进行padding
			sequences_padded, sequences_length = self._pad_sequences(sequences, 
				pad_token, max_length_sentence)

		if nlevels == 2:
			# 找到sequence中所有句子的所有单词中字母数最大的单词
			max_length_word = max([max(map(lambda x: len(x), seq)) for seq in sequences])
			print max_length_word
			sequences_padded, sequences_length = [], []
			for seq in sequences:
				print seq
				# 将每个句子的每个词都进行填充
				sp, sl = self._pad_sequences(seq, pad_token, max_length_word)
				print sp, sl
				# 每个句子的字母的表示
				sequences_padded += [sp]
				# 每个句子的字母的长度
				sequences_length += [sl]
			# 然后对句子进行填充
			# batch中最大长度的句子
			max_length_sentence = max(map(lambda x : len(x), sequences))
			# 填充的时候用[0,0,0,0,0]用字母向量进行填充
			sequences_padded, _ = self._pad_sequences(sequences_padded, 
				[pad_token] * max_length_word, max_length_sentence)
			# 得到句子的每个单词的字母的长度 (batch, max_length_sentence, letter_length)
			sequences_length, _ = self._pad_sequences(sequences_length, 0, max_length_sentence)



		return sequences_padded, sequences_length



	def _pad_sequences(self, sequences, pad_token, max_length):
		'''
		对sequences进行填充
		'''
		sequences_padded, sequences_lengths = [], []
		for sequence in sequences:
			sequence = list(sequence)
			# 获取句子长度
			sequences_lengths += [min(len(sequence), max_length)]
			# 进行填充
			sequence = sequence[:max_length] + [pad_token] * max(max_length - len(sequence), 0)
			sequences_padded += [sequence]
		return sequences_padded, sequences_lengths
		


	def get_feed_dict(self, words, labels = None, lr = None, dropout = None):
		'''
		Args:
			words: list of sentences. A sentences is a list of ids of 
			a list of words. A word is a list of ids
			labels: list of ids
			lr: learning rate
			dropout: keep prob
		'''
		if self.config['use_chars']:
			# char_ids (sent1, sent2, ..., sentn)
			# senti ([char1, char2, char3], [char1, char2, char3], ..., [cahr1, char2, char3])
			# word_ids ((word1, word2, word3),(word1, word2, word3), ... )
			char_ids, word_ids = zip(*words)
			word_ids, sequence_lengths = self.pad_sequences(word_ids, 0)
			# print word_ids
			# print sequence_lengths
			# print char_ids
			char_ids, word_lengths = self.pad_sequences(char_ids, pad_token=0, nlevels = 2)
		else:
			word_ids, sequence_lengths = self.pad_sequences(words, 0)

		feed = {
			self.word_ids: word_ids, # 句子的词的id表示(batch, max_length_sentence)
			self.sequence_lengths: sequence_lengths # 句子的词的长度(batch, )
		}

		if self.config['use_chars']:
			feed[self.char_ids] = char_ids # 句子的词的字母的id表示(batch, max_length_sentence, max_length_word)
			feed[self.word_lengths] = word_lengths # 句子的字母的长度(batch, max_length_sentence,)
		
		if labels is not None:
			labels, _ = self.pad_sequences(labels, 0)
			feed[self.labels] = labels

		if lr is not None:
			feed[self.lr] = lr
		if dropout is not None:
			feed[self.dropout] = dropout
		
		return feed, sequence_lengths

	def run_evaluate(self, test):
		'''
		在测试集上运行，并且统计结果，包括precision/recall/accuracy/f1
		Args:
			test: 一个dataset instance
		Returns:
			metrics: dict metrics['acc'] = 98.4
		'''
		accs = []
		correct_preds, total_correct, total_preds = 0., 0., 0.
		for words, labels in test.get_minibatch(self.config['batch_size']):
			# predict_batch
			# shape = (batch size, max_length_sentence)
			# shape = (batch size,)
			labels_pred, sequence_lengths = self.predict_batch(words)

			for lab, lab_pred, length in zip(labels, labels_pred, sequence_lengths):
				# 取每一句话，长度为length
				# 正确label
				lab = lab[:length]
				# 预测得到的label
				lab_pred = lab_pred[:length]

				# 预测正确的个数
				accs += [a == b for (a, b) in zip(lab, lab_pred)]

				lab_chunks = set(self.get_chunks(lab, self.vocab_tags))

				lab_pred_chunks = set(self.get_chunks(lab_pred, self.vocab_tags))

				# 记录正确的chunk数量
				correct_preds += len(lab_chunks & lab_pred_chunks)
				# 预测出的chunk数量
				total_preds += len(lab_pred_chunks)
				# 正确的chunk数量
				total_correct += len(lab_chunks)

		# 计算precision，预测出的chunk中有多少是正确的
		p = correct_preds / total_preds if correct_preds > 0 else 0
		# 计算recall，预测正确的chunk占了所有chunk的数量
		r = correct_preds / total_correct if correct_preds > 0 else 0
		# 计算f1
		f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
		# 计算accuracy，用预测对的词的比例来进行表示
		acc = np.mean(accs)

		# 返回结果
		return {
			'acc': 100 * acc,
			'f1': 100 * f1
		}



	def predict_batch(self, words):
		'''
		对一个batch进行预测，并返回预测结果
		Args:
			words: list of sentences
		Returns:
			labels_pred: list of labels for each sentence
		'''
		fd, sequence_lengths = self.get_feed_dict(words, dropout = 1.0)
		if self.config['use_crf']:
			viterbi_sequences = []
			logits, trans_params = self.sess.run(
				[self.logits, self.trans_params],
				feed_dict = fd
				)
			for logit, sequence_length in zip(logits, sequence_lengths):
				logits = logit[:sequence_length]
				viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(
					logit, trans_params)
				viterbi_sequences += [viterbi_seq]

			return viterbi_sequences, sequence_lengths

		else:
			# shape = (batch size, max_length_sentence)
			labels_pred = self.sess.run(self.labels_pred, feed_dict = fd)

			return labels_pred, sequence_lengths


	def get_chunks(self, seq, tags):
		'''
		给定一个序列的tags，将其中的entity和位置取出来
		Args:
			seq: [4,4,0,0,1,2...] 一个句子的label
			tags: dict['I-LOC'] = 2
		Returns:
			list of (chunk_type, chunk_start, chunk_end)

		Examples:
			seq = [4, 5, 0, 3]
			tags = {
			'B-PER' : 4,
			'I-PER' : 5,
			'B-LOC' : 3
			'O' : 0
			}

			Returns:
				chunks = [
					('PER', 0, 2),
					('LOC', 3, 4)
				]
		'''
		idx_to_tag = {idx : tag for tag, idx in tags.items()}
		chunks = []

		# 表示当前的chunk的起点和类型
		chunk_start, chunk_type = None, None
		print seq

		for i, tag_idx in enumerate(seq):
			# 如果不是entity的一部分
			if tag_idx == tags['O']:
				# 如果chunk_type不是None，那么就是一个entity的结束
				if chunk_type != None:
					chunk = (chunk_type, chunk_start, i)
					chunks.append(chunk)
					chunk_start, chunk_type = None, None
				# 如果chunk_type是None，那么就不需要处理
				else:
					pass
			# 如果是BI
			else:
				tag = idx_to_tag[tag_idx]
				# 如果是B
				if tag[0] == 'B':
					# 如果前面有entity，那么这个entity就完成了
					if chunk_type != None:
						chunk = (chunk_type, chunk_start, i)
						chunks.append(chunk)
						chunk_start, chunk_type = None, None

					# 记录开始
					chunk_start = i
					chunk_type = tag[2:]

				# 如果是I
				else:
					if chunk_type != None:
						# 如果chunk_type发生了变化，例如(B-PER, I-PER, B-LOC)，那么就需要将(B-PER, I-PER)归类为chunk
						if chunk_type != tag[2:]:
							chunk = (chunk_type, chunk_start, i)
							chunks.append(chunk)
							chunk_start, chunk_type = None, None
		
		# 处理可能存在的最后一个未结尾的chunk
		if chunk_type != None:
			chunk = (chunk_type, chunk_start, i + 1)
			chunks.append(chunk)
		return chunks

if __name__ == '__main__':
	model = ner_model(None, None, [None,None,None], None)
	seq = [4, 4, 5, 0, 3, 5]
	tags = {
			'B-PER' : 4,
			'I-PER' : 5,
			'B-LOC' : 3,
			'O' : 0
			}
	print model.get_chunks(seq, tags)














