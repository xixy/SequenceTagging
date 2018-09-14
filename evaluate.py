#coding=utf-8

from Configure import *
from model.Dataset import *
from model.Vocab_Util import *
from model.log_util import *
from model.ner_model import *

def main():
	'''
	构造测试集，并用restore之前的模型来进行evaluate
	'''
	# 读取配置文件
	config = Configure()
	# 设置logger
	logger = get_logger(config['path_log'])
	# 读取词典
	vocab_util = Vocab_Util()
	# dict[word] = idx
	vocab_words = vocab_util.load_vocab(config['word_vocab_file'])
	# dict[char] = idx
	vocab_chars = vocab_util.load_vocab(config['char_vocab_file'])
	# dict[tag] = idx
	vocab_tags = vocab_util.load_vocab(config['tag_vocab_file'])
	# 将词典封装给模型
	vocabs = [vocab_words, vocab_chars, vocab_tags]

	embeddings = vocab_util.get_trimmed_glove_vectors(config['trimmed_file'])

	# 对数据进行处理
	processing_word = get_processing_word(vocab_words = vocab_words, vocab_chars = vocab_chars, 
		lowercase = True, chars = config['use_chars'], allow_unk = True)
	processing_tag = get_processing_word(vocab_words = vocab_tags, lowercase = False, allow_unk = False)

	# 得到训练数据
	test_dataset = Dataset(filename = config['test_data'], 
		max_iter = None, processing_word = processing_word, processing_tag = processing_tag)	

	# 构造模型进行训练
	model = ner_model(config,logger,vocabs,embeddings)
	model.build()
	model.restore_session()
	model.evaluate(test_dataset)

if __name__ == '__main__':
	main()