#coding=utf-8
import time
import sys
import logging

def get_logger(filename):
	'''
	得到一个logger instance，写入filename中
	Args:
		filename: path to log.txt
	Returns:
		logger
	'''
	logger = logging.getLogger('logger')
	logger.setLevel(logging.DEBUG)
	logging.basicConfig(format='%(message)s', level = logging.DEBUG)
	handler = logging.FileHandler(filename)
	handler.setLevel(logging.DEBUG)
	handler.setFormatter(logging.Formatter(
		'%(asctime)s:%(levelname)s: %(message)s'))
	logging.getLogger().addHandler(handler)
	return logger