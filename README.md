# SequenceTagging
自然语言处理中的序列标注实现，包括
1. glove词向量处理
2. vocab生成
3. Bi-LSTM + CRF 模型
4. early stop
5. learning rate decay

运行方式：
1. make glove
	下载glove 向量文件，并且进行解压
2. make data
	处理train/dev/test数据，构建vocab和trimmed_vector文件
3. make train
	构建模型，并进行训练，这里采用了学习率指数递减和early stop机制
4. make evaluate
	用训练得到的模型对test数据集进行测试

配置文件请见 config.cfg