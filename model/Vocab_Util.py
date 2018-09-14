#coding=utf-8

import numpy as np
import os
from Dataset import *


# shared global variables to be imported from model also
UNK = "$UNK$"
NUM = "$NUM$"
NONE = "O"

class Vocab_Util(object):
	"""用来做词典操作"""
	def get_vocabs_from_datasets(self, datasets):
		'''
		从Dataset中获取词典
		Args:
			datasets:[dataset]多个Dataset对象的集合
		Returns:
			该dataset集合中的所有单词
		'''
		vocab_words = set()
		vocab_tags = set()
		for dataset in datasets:
			for words, tags in dataset:
				# 进行更新添加
				vocab_words.update(words)
				vocab_tags.update(tags)
		return vocab_words, vocab_tags

	def get_vocabs_from_glove(self, filename):
		'''
		从glove向量中获取vocab
		Args:
			filename: glove vector file路径
		Return:
			set() of words
		'''
		vocab = set()
		with open(filename) as f:
			for line in f:
				word = line.strip().split(' ')[0]
				vocab.add(word)
		return vocab

	def write_vocab(self, vocab, filename):
		'''
		将vocab写入到文件中，格式为一行一个词
		Args:
			vocab:iterable that yields word
			filename: 保存文件路径
		'''
		with open(filename, 'w') as f:
			for i, word in enumerate(vocab):
				if i != len(vocab) - 1:
					f.write(word + '\n')
				else:
					f.write(word)

	def load_vocab(self, filename):
		'''
		从vocab文件中加载vocab
		Args:
			filename: 保存文件路径
		Return:
			vocab: dict[word] = index
		'''
		vocab = dict()
		idx = 0
		with open(filename) as f:
			for line in f:
				word = line.strip()
				vocab[word] = idx
				idx += 1
		return vocab

	def export_trimmed_glove_vectors(self, vocab, glove_filename, trimmed_filename, dim):
		'''
		将vocab中的词从glove vector中取出来，并存储到trimed_filename文件中
		Args:
			vocab: dictionary vocab[word] = index
			glove_filename: glove vector文件路径
			trimmed_filename: 存储np matrix的路径
			dim: embedding的维度
		'''
		embeddings = np.zeros([len(vocab), dim])
		with open(glove_filename) as f:
			for line in f:
				line = line.strip().split(' ')
				word = line[0]
				embedding = [float(x) for x in line[1:]]
				if word in vocab:
					word_idx = vocab[word]
					embeddings[word_idx] = np.asarray(embedding)

		np.savez_compressed(trimmed_filename, embeddings = embeddings)

	def get_trimmed_glove_vectors(self, trimmed_filename):
		'''
		加载trimmed glove vectors
		Args:
			trimmed_filename: 存储np matrix的路径
		Returns:
			matrix of embeddings (instance of np array)
		'''
		with np.load(trimmed_filename) as data:
			return data['embeddings']


	def get_char_vocab_from_datasets(self, datasets):
		'''
		从Dataset中获取字母vocab
		Args:
			datasets:[dataset]多个Dataset对象的集合
		Returns:
			该dataset集合中的所有字母
		'''
		vocab = set()
		for words, tags in datasets:
			for word in words:
				vocab.update(word)
		return vocab


if __name__ == '__main__':
	processing_word = get_processing_word(lowercase=True)
	dataset = Dataset('../data/test.txt', processing_word = processing_word)
	vocab = Vocab_Util()
	print vocab.get_vocabs([dataset])


