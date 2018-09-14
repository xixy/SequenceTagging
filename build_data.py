#coding=utf-8
from model.Dataset import *

from Configure import *
from model.Vocab_Util import *

def main():
	'''
	完成数据的预处理
	'''
	configure = Configure('./config.cfg')
	
	processing_word = get_processing_word(lowercase=True)

	# 构造dataset
	train_dataset = Dataset(configure['train_data'], processing_word = processing_word)
	dev_dataset = Dataset(configure['train_data'], processing_word = processing_word)
	test_dataset = Dataset(configure['train_data'], processing_word = processing_word)

	# 构造word和tag的vocab
	vocab_util = Vocab_Util()
	vocab_words, vocab_tags = vocab_util.get_vocabs_from_datasets([train_dataset, dev_dataset, test_dataset])
	# 构造词向量中的词
	vocab_glove = vocab_util.get_vocabs_from_glove(configure['glove_file'])

	# 取交集，同时出现在词向量词典和数据集中的词
	vocab_words = vocab_words & vocab_glove
	# 加入UNK和数字NUM
	vocab_words.add(UNK)
	vocab_words.add(NUM)

	# 保存单词和tag的vocab文件
	vocab_util.write_vocab(vocab_words, configure['word_vocab_file'])
	vocab_util.write_vocab(vocab_tags, configure['tag_vocab_file'])

	# 获取Trim Glove Vectors，并存储
	vocab = vocab_util.load_vocab(configure['word_vocab_file'])
	vocab_util.export_trimmed_glove_vectors(vocab, configure['glove_file'], 
		configure['trimmed_file'], configure['word_embedding_dim'])
	
	# 构造char vocab, 并且进行存储
	train_dataset = Dataset(configure['train_data'])
	vocab_chars = vocab_util.get_char_vocab_from_datasets(train_dataset)
	vocab_util.write_vocab(vocab_chars, configure['char_vocab_file'])





	# 保存vocab


if __name__ == '__main__':
	main()




