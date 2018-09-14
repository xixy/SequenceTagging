#coding=utf-8

class Dataset(object):
	"""用来处理数据的类"""
	def __init__(self, filename, max_iter = None, processing_word = None, processing_tag = None):
		'''
		Args:
			filename: 该Dataset对应的数据文件路径 ./data/train.txt
			max_iter: max number of sentences to yield from this dataset
			processing_word: 对词进行处理的函数
			processing_tag: 对tag进行处理的函数
		'''
		self.filename = filename
		self.max_iter = max_iter
		self.length = None # 数据集中的句子数量
		self.processing_word = processing_word
		self.processing_tag = processing_tag

	def __iter__(self):
		'''
		支持iteration的遍历操作
		'''
		count = 0
		with open(self.filename) as f:
			words, tags = [], []
			for line in f:
				line = line.strip()
				if (len(line) == 0 or line.startswith("-DOCSTART-")):
					# 如果是新的一行，表示句子结束
					if len(words) != 0:
						count += 1
						# 如果超过了
						if self.max_iter is not None and count > self.max_iter:
							break
						yield words, tags
						words, tags = [], []
					
				else:
					elems = line.split(' ')
					word, tag = elems[0], elems[1]
					# 进行预处理
					if self.processing_word is not None:
						word = self.processing_word(word)
					if self.processing_tag is not None:
						tag = self.processing_tag(tag)

					words += [word]
					tags += [tag]

	def __len__(self):
		'''
		支持len操作来查看句子数量
		'''
		# 如果没初始化，然后再进行统计
		if self.length is None:
			self.length = 0
			for _ in self:
				self.length += 1
		return self.length

	def get_minibatch(self, batch_size):
		'''
		从数据集中获取minibatch
		'''
		x_batch, y_batch = [], []
		for (x, y) in self:
			if len(x_batch) == batch_size:
				yield x_batch, y_batch
				x_batch, y_batch = [], []

			# 如果是([char_ids], word_id)的形式
			if type(x[0]) == tuple:
				x = zip(*x)

			x_batch += [x]
			y_batch += [y]
		if len(x_batch) != 0:
			yield x_batch, y_batch



def get_processing_word(vocab_words = None, vocab_chars = None, 
	lowercase = False, chars = False, allow_unk = True):
    """Return lambda function that transform a word (string) into list,
    or tuple of (list, id) of int corresponding to the ids of the word and
    its corresponding characters.

    Args:
        vocab: dict[word] = idx
        vocab_chars: dict[char] = idx
        lowercase: 是否将单词进行小写化
        chars: 是否要返回单词id列表
        allow_unk: 是否将不在词典中的单词作为UNK

    Returns:
        f("cat") = ([12, 4, 32], 12345)
                 = (list of char ids, word id)

    """
    def f(word):

    	# 首先转化为chars list
    	if vocab_chars is not None and chars:
    		char_ids = []
    		for char in word:
    			# 如果字母不在的话，就忽略，基本上很少这种情况
    			if char in vocab_chars:
    				char_ids.append(vocab_chars[char])
    	# 对单词进行处理
    	if lowercase:
    		word = word.lower()
    	if word.isdigit():
    		word = NUM

    	# 得到单词的id
    	if vocab_words is not None:
    		if word in vocab_words:
    			word = vocab_words[word]
    		else:
    			if allow_unk:
    				word = vocab_words[UNK]
    			else:
    				raise Exception('出现了不存在词典中的词，请检查是否正确' + word)
    	# 返回char ids, word
    	if vocab_chars is not None and chars == True:
    		return char_ids, word
    	else:
    		return word
    return f

if __name__ == '__main__':
	processing_word = get_processing_word(lowercase=True)
	dataset = Dataset('../data/test.txt', processing_word = processing_word)

	for data in dataset:
		print data
	print len(dataset)
	for x_batch , y_batch in dataset.get_minibatch(5):
		print x_batch
		print y_batch
	# 	print data

