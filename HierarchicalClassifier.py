#!/usr/bin/python2

import arff
import numpy as np
import sys
import getopt
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import weka.core.jvm as jvm
from weka.core.converters import Loader, ndarray_to_instances
from weka.core.dataset import Instances, Attribute
from weka.classifiers import Classifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import CategoricalEncoder
from sklearn.impute import SimpleImputer
import progressbar
import networkx as nx
import psutil
from threading import Thread
from threading import Semaphore

sem = Semaphore(5)

class Tree(object):
	def __init__(self):

		# Position info
		self.parent = None
		self.children = {}
		
		# Node info
		self.tag = 'ROOT'
		self.level = 0
		self.children_tags = []
		self.children_number = 0
		
		# Classification info
		self.features_index = []
		self.train_index = []
		self.test_index = []

		self.test_index_all = []

		# Configuration info
		self.features_number = 0
		self.packets_number = 0
		self.classifier_name = 'srf'

class SklearnWekaWrapper(object):

	def __init__(self, class_name, options=None):

		if options is not None:
			self._classifier = Classifier(classname=class_name, options=[option for option in options.split()])
		else:
			self._classifier = Classifier(classname=class_name)

	def fit(self, training_set, ground_through):

		self.ground_through = ground_through

		training_set = self._sklearn2weka(training_set, self.ground_through)
		training_set.class_is_last()

		self._classifier.build_classifier(training_set)

	def predict(self, testing_set):

		testing_set = self._sklearn2weka(testing_set, self.ground_through)
		testing_set.class_is_last()

		preds = []
		for index, inst in enumerate(testing_set):
			pred = self._classifier.classify_instance(inst)
			preds.append(pred)

		preds = np.vectorize(self._dict.get)(preds)

		return np.array(preds)

	def predict_proba(self, testing_set):

		testing_set = self._sklearn2weka(testing_set, self.ground_through)
		testing_set.class_is_last()

		dists = []
		for index, inst in enumerate(testing_set):
			dist = self._classifier.distribution_for_instance(inst)
			dists.append(dist)

		return np.array(dists)

	def _sklearn2weka(self, features, labels=None):

		encoder = CategoricalEncoder(encoding='ordinal')
		labels_nominal = encoder.fit_transform(np.array(labels).reshape(-1, 1))

		if not hasattr(self, 'dict') and labels is not None:

			dict = {}

			for label, nominal in zip(labels, labels_nominal):
				if nominal.item(0) not in dict:
					dict[nominal.item(0)] = label

			self._dict = dict

		labels_column = np.reshape(labels_nominal,[labels_nominal.shape[0], 1])

		weka_dataset = ndarray_to_instances(np.ascontiguousarray(features, dtype=np.float_), 'weka_dataset')
		weka_dataset.insert_attribute(Attribute.create_nominal('tag', [str(float(i)) for i in range(len(self._dict))]), features.shape[1])

		if labels is not None:
			for index, inst in enumerate(weka_dataset):
				inst.set_value(features.shape[1], labels_column[index])
				weka_dataset.set_instance(index,inst)

		return weka_dataset

class HierarchicalClassifier(object):
	def __init__(self,input_file,levels_number,features_number,packets_number,classifier_name):

		self.input_file = input_file
		self.levels_number = levels_number
		self.features_number = features_number
		self.packets_number = packets_number
		self.classifier_name = classifier_name
		self.has_config = False

	def set_config(self, config_name, config):
		self.has_config = True
		self.config_name = config_name
		self.config = config

	def kfold_validation(self, k=10):

		sem.acquire()

		available_ram = psutil.virtual_memory()[1]
		available_ram = int(int(available_ram) * .9 * 1e-9)

		if available_ram > 5:
			jvm.start(max_heap_size='5g')
		else:
			jvm.start(max_heap_size=str(available_ram)+'g')

		###

		print('\nCaricando '+self.input_file+' con opts -f'+str(self.features_number)+' -c'+self.classifier_name+'\n')
		# load .arff file
		dataset = arff.load(open(self.input_file, 'r'))
		data = np.array(dataset['data'])

		self.features_names = [x[0] for x in dataset['attributes']]

		self.attributes_number = data.shape[1]
		self.dataset_features_number = self.attributes_number - self.levels_number

		# Factorization of Nominal features_index
		encoder = CategoricalEncoder(encoding='ordinal')
		nominal_features_index = [i for i in range(len(dataset['attributes'][:-self.levels_number])) if dataset['attributes'][i][1] != u'NUMERIC']
		if len(nominal_features_index) > 0:
			data[:, nominal_features_index] = encoder.fit_transform(
				data[:, nominal_features_index])

		# Impute missing value by fitting over training set and transforming both sets
		imp = SimpleImputer(missing_values='NaN', strategy='most_frequent')
		data[:, :self.dataset_features_number] = imp.fit_transform(data[:, :self.dataset_features_number])

		classifiers_per_fold = []
		oracles_per_fold = []
		predictions_per_fold = []
		predictions_per_fold_all = []

		print('\n***\nStart testing with '+str(k)+'Fold cross-validation -f'+str(self.features_number)+' -c'+self.classifier_name+'\n***\n')

		bar = progressbar.ProgressBar(maxval=k, widgets=[progressbar.Bar(
			'=', '[', ']'), ' ', progressbar.Percentage()])
		bar.start()

		skf = StratifiedKFold(n_splits=k, shuffle=True)
		bar_cnt = 0

		for train_index, test_index in skf.split(data, data[:,self.attributes_number-1]):

			self.classifiers = []

			self.training_set = data[train_index, :self.dataset_features_number]
			self.testing_set = data[test_index, :self.dataset_features_number]
			self.ground_through = data[train_index, self.dataset_features_number:]
			self.oracle = data[test_index, self.dataset_features_number:]
			self.prediction = np.ndarray(shape=[len(test_index),self.levels_number],dtype='<U24')
			self.prediction_all = np.ndarray(shape=[len(test_index),self.levels_number],dtype='<U24')

			root = Tree()

			root.train_index = [i for i in range(self.training_set.shape[0])]
			root.test_index = [i for i in range(self.testing_set.shape[0])]
			root.test_index_all = root.test_index
			root.children_tags = list(set(self.ground_through[root.train_index, root.level]))
			root.children_number = len(root.children_tags)

			if self.has_config:
				if 'f' in config[root.tag + '_' + str(root.level + 1)]:
					root.features_number = config[root.tag + '_' + str(root.level + 1)]['f']
				elif 'p' in config[root.tag + '_' + str(root.level + 1)]:
					root.packets_number = config[root.tag + '_' + str(root.level + 1)]['p']
				root.classifier_name = config[root.tag + '_' + str(root.level + 1)]['c']

				print('config','tag',root.tag,'level',root.level,'f',root.features_number,'c',root.classifier_name)
			else:
				root.features_number = self.features_number
				root.packets_number = self.packets_number
				root.classifier_name = self.classifier_name

			self.classifiers.append(root)

			if root.children_number > 1:

				classifier_to_call = getattr(self, supported_classifiers[root.classifier_name])
				classifier_to_call(node=root)

			else:

				self.unary_class_results_inferring(root)

			# Creating hierarchy recursively
			if root.level < self.levels_number-1 and root.children_number > 0:
				self.recursive(root)

			classifiers_per_fold.append(self.classifiers)

			oracles_per_fold.append(self.oracle)
			predictions_per_fold.append(self.prediction)
			predictions_per_fold_all.append(self.prediction_all)

			bar_cnt += 1
			bar.update(bar_cnt)

		bar.finish()

		folder_discriminator = self.classifier_name

		if self.has_config:
			folder_discriminator = self.config_name

		material_folder = './data_'+folder_discriminator+'/material/'

		if not os.path.exists('./data_'+folder_discriminator):
			os.makedirs('./data_'+folder_discriminator)
			os.makedirs(material_folder)
		elif not os.path.exists(material_folder):
			os.makedirs(material_folder)

		type_discr = 'flow'
		feat_discr = '_f_' + str(self.features_number)

		if not self.has_config and self.packets_number != 0:
			type_discr = 'early'
			feat_discr = '_p_' + str(self.packets_number)
		elif self.has_config:
			if 'p' in self.config:
				type_discr = 'early'
			feat_discr = '_c_' + self.config_name

		material_features_folder = './data_'+folder_discriminator+'/material/features/'

		if not os.path.exists(material_folder):
			os.makedirs(material_folder)
			os.makedirs(material_features_folder)
		elif not os.path.exists(material_features_folder):
			os.makedirs(material_features_folder)

		for i in range(self.levels_number):

			file = open(material_folder + 'multi_' + type_discr + '_level_' + str(i+1) + feat_discr + '.dat', 'w+')
			file.close()

			for j in range(k):

				file = open(material_folder + 'multi_' + type_discr + '_level_' + str(i+1) + feat_discr + '.dat', 'a')

				file.write('@fold\n')
				for o, p in zip(oracles_per_fold[j][:,i], predictions_per_fold[j][:,i]):
					file.write(str(o)+' '+str(p)+'\n')

				file.close()

		# Inferring NW metrics per classifier

		for classifier in classifiers_per_fold[0]:

			file = open(material_folder + 'multi_' + type_discr + '_level_' + str(classifier.level+1) + feat_discr + '_tag_' + str(classifier.tag) + '.dat', 'w+')
			file.close()

			file = open(material_folder + 'multi_' + type_discr + '_level_' + str(classifier.level+1) + feat_discr + '_tag_' + str(classifier.tag) + '_all.dat', 'w+')
			file.close()

			file = open(material_features_folder + 'multi_' + type_discr + '_level_' + str(classifier.level+1) + feat_discr + '_tag_' + str(classifier.tag) + '_features.dat', 'w+')
			file.close()

		for fold_n, classifiers in enumerate(classifiers_per_fold):

			for classifier in classifiers:

				file = open(material_folder + 'multi_' + type_discr + '_level_' + str(classifier.level+1) + feat_discr + '_tag_' + str(classifier.tag) + '.dat', 'a')

				if classifier.level > 0:
					index = []

					for pred_n, prediction in enumerate(predictions_per_fold[fold_n][classifier.test_index, classifier.level-1]):
						if prediction == oracles_per_fold[fold_n][classifier.test_index[pred_n], classifier.level-1]:
							index.append(classifier.test_index[pred_n])

					prediction_nw = predictions_per_fold[fold_n][index, classifier.level]
					oracle_nw = oracles_per_fold[fold_n][index, classifier.level]
				else:
					prediction_nw = predictions_per_fold[fold_n][classifier.test_index, classifier.level]
					oracle_nw = oracles_per_fold[fold_n][classifier.test_index, classifier.level]

				file.write('@fold\n')
				for o, p in zip(oracle_nw, prediction_nw):
						file.write(str(o)+' '+str(p)+'\n')

				file.close()

				file = open(material_folder + 'multi_' + type_discr + '_level_' + str(classifier.level+1) + feat_discr + '_tag_' + str(classifier.tag) + '_all.dat', 'a')

				if classifier.level > 0:
					index = []

					for pred_n, prediction in enumerate(predictions_per_fold_all[fold_n][classifier.test_index, classifier.level-1]):
						if prediction == oracles_per_fold[fold_n][classifier.test_index[pred_n], classifier.level-1]:
							index.append(classifier.test_index[pred_n])

					prediction_all = predictions_per_fold_all[fold_n][index, classifier.level]
					oracle_all = oracles_per_fold[fold_n][index, classifier.level]
				else:
					prediction_all = predictions_per_fold_all[fold_n][classifier.test_index_all, classifier.level]
					oracle_all = oracles_per_fold_all[fold_n][classifier.test_index_all, classifier.level]

				file.write('@fold\n')
				for o, p in zip(oracle_all, prediction_all):
						file.write(str(o)+' '+str(p)+'\n')

				file.close()

				file = open(material_features_folder + 'multi_' + type_discr + '_level_' + str(classifier.level+1) + feat_discr + '_tag_' + str(classifier.tag) + '_features.dat', 'a')

				file.write('@fold\n')
				file.write(self.features_names[classifier.features_index[0]])

				for feature_index in classifier.features_index[1:]:
					file.write(','+self.features_names[feature_index])

				file.write('\n')

				file.close()

		graph_folder = './data_'+folder_discriminator+'/graph/'

		if not os.path.exists('./data_'+folder_discriminator):
			os.makedirs('./data_'+folder_discriminator)
			os.makedirs(graph_folder)
		elif not os.path.exists(graph_folder):
			os.makedirs(graph_folder)

		# Graph plot
		G = nx.DiGraph()
		for info in classifiers_per_fold[0]:
			G.add_node(str(info.level)+' '+info.tag, level=info.level,
					   tag=info.tag, children_tags=info.children_tags)
		for node_parent, data_parent in G.nodes.items():
			for node_child, data_child in G.nodes.items():
				if data_child['level']-data_parent['level'] == 1 and any(data_child['tag'] in s for s in data_parent['children_tags']):
					G.add_edge(node_parent, node_child)
		nx.write_gpickle(G, graph_folder+'multi_' + type_discr + feat_discr +'_graph.gml')

		###

		jvm.stop()

		sem.release()

	def features_selection(self,node):
		features_index = []

		if node.features_number != 0 and node.features_number != self.dataset_features_number:

			# print('\n***\nFeature Selection for Classifier ' + node.tag + ' Level ' + str(node.level) + '\n***\n')

			selector = SelectKBest(mutual_info_classif, k=node.features_number)
			training_set_selected = selector.fit_transform(
				self.training_set[node.train_index, :self.dataset_features_number], self.ground_through[node.train_index, node.level])
			training_set_reconstr = selector.inverse_transform(
				training_set_selected)

			i0 = 0
			i1 = 0
			while i0 < node.features_number:
				if np.array_equal(training_set_selected[:, i0], training_set_reconstr[:, i1]):
					features_index.append(i1)
					i0 += 1
				i1 += 1
		else:
			if node.packets_number == 0:
				features_index = [i for i in range(self.dataset_features_number)]
			else:
				features_index = np.r_[0:node.packets_number, self.dataset_features_number/2:self.dataset_features_number/2+node.packets_number]
			
		return features_index

	def recursive(self,parent):

		for i in range(parent.children_number):
			child = Tree()
			self.classifiers.append(child)

			child.level = parent.level+1
			child.tag = parent.children_tags[i]
			child.parent = parent
			
			child.train_index = [index for index in parent.train_index if self.ground_through[index, parent.level] == child.tag]
			child.test_index = [index for index in parent.test_index if self.prediction[index, parent.level] == child.tag]

			child.test_index_all = [index for index in parent.test_index_all if self.ground_through[index, parent.level] == child.tag]
			child.children_tags = list(set(self.ground_through[child.train_index, child.level]))

			child.children_number = len(child.children_tags)

			if self.has_config:
				if 'f' in config[child.tag + '_' + str(child.level + 1)]:
					child.features_number = config[child.tag + '_' + str(child.level + 1)]['f']
				elif 'p' in config[child.tag + '_' + str(child.level + 1)]:
					child.packets_number = config[child.tag + '_' + str(child.level + 1)]['p']
				child.classifier_name = config[child.tag + '_' + str(child.level + 1)]['c']
				print('config','tag',child.tag,'level',child.level,'f',child.features_number,'c',child.classifier_name)
			else:
				child.features_number = self.features_number
				child.packets_number = self.packets_number
				child.classifier_name = self.classifier_name

			# print(self.prediction[parent.test_index])

			# print(child.tag, child.level)
			# print(len(child.train_index))

			self.classifiers[self.classifiers.index(parent)].children[child.tag]=child

			if child.children_number > 1:

				classifier_to_call = getattr(self, supported_classifiers[child.classifier_name])
				classifier_to_call(node=child)

			else:

				self.unary_class_results_inferring(child)

			if child.level < self.levels_number-1 and child.children_number > 0:
				self.recursive(child)

	def unary_class_results_inferring(self, node):

		for index, pred in zip(node.test_index, self.prediction[node.test_index, node.level]):

			self.prediction[index, node.level] = node.tag

		node.features_index = node.parent.features_index

	def Sklearn_RandomForest(self, node):

		# Instantation
		classifier = RandomForestClassifier(n_estimators=100, n_jobs=-1)

		# Features selection
		node.features_index = self.features_selection(node)

		self.train(node, classifier)
		self.test(node, classifier)

		self.test_all(node, classifier)

	def Sklearn_CART(self, node):

		# Instantation
		classifier = DecisionTreeClassifier()

		# Features selection
		node.features_index = self.features_selection(node)

		self.train(node, classifier)
		self.test(node, classifier)

		self.test_all(node, classifier)

	def Weka_NaiveBayes(self, node):

		# Instantation
		classifier = SklearnWekaWrapper(class_name='weka.classifiers.bayes.NaiveBayes', options='-D')

		# Features selection
		node.features_index = self.features_selection(node)

		self.train(node, classifier)
		self.test(node, classifier)

		self.test_all(node, classifier)

	def Weka_BayesNetwork(self, node):

		# Instantation
		classifier = SklearnWekaWrapper(class_name='weka.classifiers.bayes.BayesNet', options='-D -Q weka.classifiers.bayes.net.search.local.TAN -- -S BAYES -E weka.classifiers.bayes.net.estimate.SimpleEstimator -- -A 0.5')

		# Features selection
		node.features_index = self.features_selection(node)

		self.train(node, classifier)
		self.test(node, classifier)

		self.test_all(node, classifier)

	def Weka_RandomForest(self, node):

		# Instantation
		classifier = SklearnWekaWrapper(class_name='weka.classifiers.trees.RandomForest')

		# Features selection
		node.features_index = self.features_selection(node)

		self.train(node, classifier)
		self.test(node, classifier)

		self.test_all(node, classifier)

	def Weka_J48(self, node):

		# Instantation
		classifier = SklearnWekaWrapper(class_name='weka.classifiers.trees.J48')

		# Features selection
		node.features_index = self.features_selection(node)

		self.train(node, classifier)
		self.test(node, classifier)

		self.test_all(node, classifier)

	def train(self, node, classifier):

		classifier.fit(self.training_set[node.train_index][:, node.features_index], self.ground_through[node.train_index][:, node.level])

	def test(self, node, classifier):

		pred = classifier.predict(self.testing_set[node.test_index][:, node.features_index])

		for p, i in zip(pred, node.test_index):
			self.prediction[i, node.level] = p

	def test_all(self, node, classifier):

		pred = classifier.predict(self.testing_set[node.test_index_all][:, node.features_index])

		for p, i in zip(pred, node.test_index_all):
			self.prediction_all[i, node.level] = p

if __name__ == "__main__":
	os.environ["DISPLAY"] = ":0"  # used to show xming display
	np.random.seed(0)

	supported_classifiers = {
		'srf': 'Sklearn_RandomForest',
		'scr': 'Sklearn_CART',
		'wnb': 'Weka_NaiveBayes',
		'wbn': 'Weka_BayesNetwork',
		'wrf': 'Weka_RandomForest',
		'wj48': 'Weka_J48'
	}

	input_file = ''
	levels_number = 0
	features_number = 0
	packets_number = 0
	classifier_name = 'srf'

	general = False

	config_file = ''
	config = ''

	try:
		opts, args = getopt.getopt(
			sys.argv[1:], "hi:n:f:p:c:go:", "[input_file=,levels_number=,features_number=,packets_number=,classifier_name=,configuration=]")
	except getopt.GetoptError:
		print('MultilayerClassifier3.0.py -i <input_file> -n <levels_number> (-f <features_number>|-p <packets_number>) -c <classifier_name> (-g)')
		sys.exit(2)
	for opt, arg in opts:
		if opt in ("-h", "--help"):
			print('usage: MultilayerClassifier3.0.py -i <input_file> -n <levels_number> (-f <features_number>|-p <packets_number>) -c <classifier_name>')
			sys.exit()
		if opt in ("-i", "--input_file"):
			input_file = arg
		if opt in ("-n", "--levels_number"):
			levels_number = int(arg)
		if opt in ("-f", "--nfeat"):
			features_number = int(arg)
		if opt in ("-p", "--npacket"):
			packets_number = int(arg)
		if opt in ("-c", "--clf"):
			classifier_name = arg
		if opt in ("-g", "--general"):
			general = True
		if opt in ("-o", "--configuration"):
			config_file = arg

	if config_file:
		if not config_file.endswith('.json'):
			print('config file must have .json extention')
			sys.exit()

		import json

		with open(config_file) as f:
			config = json.load(f)

	else:

		if not general and (packets_number != 0 and features_number != 0 or packets_number == features_number):
			print('-f and -p option should not be used together')
			sys.exit()

		if classifier_name not in supported_classifiers:
			print('Classifier not supported\nList of available classifiers:\n')
			for sc in supported_classifiers:
				print('-c '+sc+'\t--->\t'+supported_classifiers[sc].split('_')[1]+'\t\timplemented in '+supported_classifiers[sc].split('_')[0])
			sys.exit()

	if levels_number == 0:
		print('MultilayerClassifier3.0.py -i <input_file> -n <levels_number> (-f <features_number>|-p <packets_number>) -c <classifier_name>')
		print('Number of level must be positive and non zero')
		sys.exit()

	if not input_file.endswith(".arff"):
		print('input files must be .arff')
		sys.exit()

	if not general:

		hierarchical_classifier = HierarchicalClassifier(input_file=input_file,levels_number=levels_number,features_number=features_number,packets_number=packets_number,classifier_name=classifier_name)
		if config:
			config_name = config_file.split('.')[0]
			hierarchical_classifier.set_config(config_name, config)
		hierarchical_classifier.kfold_validation(k=10)

	else:

		# jvm.start(max_heap_size=str(available_ram)+'g')

		hierarchical_classifiers = []

		for classifier_name in supported_classifiers:
			if 'w' in classifier_name:
				for features_number in range(5,74,5):
					hierarchical_classifiers.append(HierarchicalClassifier(input_file=input_file,levels_number=levels_number,features_number=features_number,packets_number=packets_number,classifier_name=classifier_name))
				hierarchical_classifiers.append(HierarchicalClassifier(input_file=input_file,levels_number=levels_number,features_number=74,packets_number=packets_number,classifier_name=classifier_name))

		threads = []

		for hierarchical_classifier in hierarchical_classifiers:
			threads.append(Thread(target=hierarchical_classifier.kfold_validation, args=(10, )))

		for thread in threads:
			thread.start()

		for thread in threads:
			thread.join()

			del thread

		# for hierarchical_classifier in hierarchical_classifiers:
		# 	hierarchical_classifier.kfold_validation(10)

		# jvm.stop()
