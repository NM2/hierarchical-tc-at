#!/usr/bin/python2

#    Implements flat classifiers employed in the paper "A Dive into the Dark Web: Hierarchical Traffic Classification of Anonymity Tools".
#
#    Copyright (C) 2018  Giampaolo Bovenzi & Antonio Montieri
#    email: traffic@unina.it, giampaolo.bovenzi@gmail.com, antonio.montieri@unina.it
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Affero General Public License as
#    published by the Free Software Foundation, either version 3 of the
#    License, or (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU Affero General Public License for more details.
#
#    You should have received a copy of the GNU Affero General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

import arff
import numpy as np
import copy
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
import psutil

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

class FlatClassifier(object):
	def __init__(self, input_file, levels_number, level_target, features_number, packets_number, classifier_name):

		self.input_file = input_file
		self.levels_number = levels_number
		self.level_target = level_target
		self.features_number = features_number
		self.packets_number = packets_number
		self.classifier_name = classifier_name
		self.tag_under_test = level_target-1

	def kfold_validation(self, k=10):

		available_ram = psutil.virtual_memory()[1]
		available_ram = int(int(available_ram) * .9 * 1e-9)

		if available_ram > 5:
			jvm.start(max_heap_size='5g')
		else:
			print('Seem your machine has less than 5 GB amount of RAM available:\n')
			print('cannot start jvm.')
			sys.exit()

		###

		print('\nCaricando '+self.input_file+' con opts -f'+str(self.features_number)+' -c'+self.classifier_name+'\n')
		# load .arff file
		dataset = arff.load(open(input_file, 'r'))
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

		prediction = []
		probability = []
		oracle = []

		print('\n***\nStart testing with ' + str(k)+'Fold cross-validation -f'+str(self.features_number)+' -c'+self.classifier_name+'\n***\n')

		bar = progressbar.ProgressBar(maxval=k, widgets=[progressbar.Bar(
			'=', '[', ']'), ' ', progressbar.Percentage()])
		bar.start()

		temp_metrics = []

		skf = StratifiedKFold(n_splits=k, shuffle=True)
		bar_cnt = 0
		for train_index, test_index in skf.split(data, data[:, self.dataset_features_number + self.tag_under_test]):

			self.training_set = data[train_index, :self.dataset_features_number]
			self.testing_set = data[test_index, :self.dataset_features_number]
			self.ground_through = data[train_index,
									   self.dataset_features_number + self.tag_under_test]
			self.oracle = data[test_index,
							   self.dataset_features_number + self.tag_under_test]
			self.prediction = np.ndarray(
				shape=[len(test_index), 1], dtype='<U24')
			self.probability = np.ndarray(
				shape=[len(test_index), len(set(self.ground_through))], dtype='<U24')

			classifier_to_call = getattr(self, supported_classifiers[self.classifier_name])
			classifier_to_call()

			prediction.append(self.prediction)
			probability.append(self.probability)
			oracle.append(self.oracle)

			bar_cnt += 1
			bar.update(bar_cnt)

		bar.finish()

		relations = []

		relations = []
		relations.append({  # Lv2:Lv1
			u'Tor': u'Tor',
			u'TorPT': u'Tor',
			u'TorApp': u'Tor',
			u'I2PApp80BW': u'I2P',
			u'I2PApp0BW': u'I2P',
			u'I2PApp': u'I2P',
			u'JonDonym': u'JonDonym'
		})

		relations.append({  # Lv3:Lv2
			u'JonDonym': u'JonDonym',
			u'I2PSNARK_App80BW': u'I2PApp80BW',
			u'IRC_App80BW': u'I2PApp80BW',
			u'Eepsites_App80BW': u'I2PApp80BW',
			u'I2PSNARK_App0BW': u'I2PApp0BW',
			u'IRC_App0BW': u'I2PApp0BW',
			u'Eepsites_App0BW': u'I2PApp0BW',
			u'I2PSNARK_App': u'I2PApp',
			u'IRC_App': u'I2PApp',
			u'Eepsites_App': u'I2PApp',
			u'ExploratoryTunnels_App': u'I2PApp',
			u'ParticipatingTunnels_App': u'I2PApp',
			u'Tor': u'Tor',
			u'Streaming': u'TorApp',
			u'Torrent': u'TorApp',
			u'Browsing': u'TorApp',
			u'Flashproxy': u'TorPT',
			u'FTE': u'TorPT',
			u'Meek': u'TorPT',
			u'Obfs3': u'TorPT',
			u'scramblesuit': u'TorPT'
		})

		oracle_inferred = []
		prediction_inferred = []

		for i in range(self.tag_under_test):
			oracle_inferred.append(list())
			prediction_inferred.append(list())

		# Infering superior levels
		for i in range(k):
			# Assign of prediction to a dummy to use this one in consecutive label swaps
			inferred_prediction = prediction[i].copy()
			inferred_oracle = oracle[i].copy()
			for j in reversed(range(self.tag_under_test)):
				inferred_oracle = np.vectorize(
					relations[j].get)(list(inferred_oracle))
				inferred_prediction = np.vectorize(
					relations[j].get)(list(inferred_prediction))
				oracle_inferred[j].append(inferred_oracle)
				prediction_inferred[j].append(inferred_prediction)
		print('\n***\nStart testing with incremental gamma threshold\n***\n')

		bar = progressbar.ProgressBar(maxval=9, widgets=[progressbar.Bar(
			'=', '[', ']'), ' ', progressbar.Percentage()])
		bar.start()

		oracle_gamma = []
		prediction_gamma = []
		classified_ratio = []

		for i in range(9):
			gamma = float(i+1)/10.0

			oracle_gamma.append(list())
			prediction_gamma.append(list())
			classified_ratio.append(list())

			for j in range(k):
				indexes = []
				p_cnt = 0
				for p in probability[j]:
					if max(p) < gamma:
						indexes.append(p_cnt)
					p_cnt += 1
				gamma_oracle = np.delete(oracle[j], [indexes])
				gamma_prediction = np.delete(prediction[j], [indexes])
				oracle_gamma[i].append(gamma_oracle)
				prediction_gamma[i].append(gamma_prediction)
				classified_ratio[i].append(
					float(len(gamma_prediction))/float(len(prediction[j])))

			bar.update(i)

		bar.finish()

		data_folder = './data_'+self.classifier_name+'/material/'

		if not os.path.exists('./data_'+self.classifier_name):
			os.makedirs('./data_'+self.classifier_name)
			os.makedirs(data_folder)
		elif not os.path.exists(data_folder):
			os.makedirs(data_folder)

		if self.packets_number != 0:
			file = open(data_folder+'flat_early_level_'+str(self.level_target) +
						'_p_'+str(self.packets_number)+'.dat', 'w+')
		else:
			file = open(data_folder+'flat_flow_level_'+str(self.level_target) +
						'_f_'+str(self.features_number)+'.dat', 'w+')

		for i in range(k):
			file.write('@fold\n')
			for o, p in zip(oracle[i], prediction[i]):
				file.write(str(o)+' '+str(p)+'\n')

		file.close()

		for i in range(self.tag_under_test):

			if self.packets_number != 0:
				file = open(data_folder+'flat_early_level_'+str(self.level_target) +
							'_p_'+str(self.packets_number)+'_inferred_'+str(i+1)+'.dat', 'w+')
			else:
				file = open(data_folder+'flat_flow_level_'+str(self.level_target) +
							'_f_'+str(self.features_number)+'_inferred_'+str(i+1)+'.dat', 'w+')

			for j in range(k):
				file.write('@fold\n')
				for o, p in zip(oracle_inferred[i][j], prediction_inferred[i][j]):
					file.write(str(o)+' '+str(p)+'\n')

			file.close()

		for i in range(9):
			if self.packets_number != 0:
				file = open(data_folder+'flat_early_level_'+str(self.level_target)+'_p_' +
							str(self.packets_number)+'_gamma_'+str(float(i+1)/10.0)+'.dat', 'w+')
			else:
				file = open(data_folder+'flat_flow_level_'+str(self.level_target)+'_f_' +
							str(self.features_number)+'_gamma_'+str(float(i+1)/10.0)+'.dat', 'w+')

			for j in range(k):
				file.write('@fold_cr\n')
				file.write(str(classified_ratio[i][j])+'\n')
				for o, p in zip(oracle_gamma[i][j], prediction_gamma[i][j]):
					file.write(str(o)+' '+str(p)+'\n')

			file.close()

		###

		jvm.stop()

	def features_selection(self):

		features_index = []

		if self.features_number != 0 and self.features_number != self.dataset_features_number:

			selector = SelectKBest(mutual_info_classif, k=self.features_number)
			training_set_selected = selector.fit_transform(
				self.training_set[:, :self.dataset_features_number], self.ground_through)
			training_set_reconstr = selector.inverse_transform(
				training_set_selected)

			i0 = 0
			i1 = 0
			while i0 < self.features_number:
				if np.array_equal(training_set_selected[:, i0], training_set_reconstr[:, i1]):
					features_index.append(i1)
					i0 += 1
				i1 += 1
		else:
			if self.packets_number == 0:
				features_index = [i for i in range(self.dataset_features_number)]
			else:
				features_index = np.r_[0:self.packets_number, self.dataset_features_number /
					2:self.dataset_features_number/2+self.packets_number]

		return features_index

	def Sklearn_RandomForest(self):

		# Instantation
		classifier = RandomForestClassifier(n_estimators=100, n_jobs=-1)

		# Features selection
		features_index = self.features_selection()

		self.train(classifier, features_index)
		self.test(classifier, features_index)

	def Sklearn_CART(self):

		# Instantation
		classifier = DecisionTreeClassifier()

		# Features selection
		features_index = self.features_selection()

		self.train(classifier, features_index)
		self.test(classifier, features_index)

	def Weka_NaiveBayes(self):

		# Instantation
		classifier = SklearnWekaWrapper(class_name='weka.classifiers.bayes.NaiveBayes', options='-D')

		# Features selection
		features_index = self.features_selection()

		self.train(classifier, features_index)
		self.test(classifier, features_index)

	def Weka_BayesNetwork(self):

		# Instantation
		classifier = SklearnWekaWrapper(class_name='weka.classifiers.bayes.BayesNet', options='-D -Q weka.classifiers.bayes.net.search.local.TAN -- -S BAYES -E weka.classifiers.bayes.net.estimate.SimpleEstimator -- -A 0.5')

		# Features selection
		features_index = self.features_selection()

		self.train(classifier, features_index)
		self.test(classifier, features_index)

	def Weka_RandomForest(self):

		# Instantation
		classifier = SklearnWekaWrapper(class_name='weka.classifiers.trees.RandomForest')

		# Features selection
		features_index = self.features_selection()

		self.train(classifier, features_index)
		self.test(classifier, features_index)

	def Weka_J48(self):

		# Instantation
		classifier = SklearnWekaWrapper(class_name='weka.classifiers.trees.J48')

		# Features selection
		features_index = self.features_selection()

		self.train(classifier, features_index)
		self.test(classifier, features_index)

	def train(self, classifier, features_index):

		classifier.fit(self.training_set[:, features_index], self.ground_through)

	def test(self, classifier, features_index):

		self.prediction = classifier.predict(
			self.testing_set[:, features_index])
		self.probability = classifier.predict_proba(
			self.testing_set[:, features_index])

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
	level_target = 0
	features_number = 0
	packets_number = 0
	classifier_name = 'srf'

	try:
		opts, args = getopt.getopt(
			sys.argv[1:], "hi:n:t:f:p:c:", "[input_file=,levels_number=,level_target=,features_number=,packets_number=,classifier=]")
	except getopt.GetoptError:
		print('FlatClassifier.py -i <input_file> -n <levels_number> -t <level_target> (-f <features_number>|-p <packets_number>) -c <classifier_name>')
		print('FlatClassifier.py -h (or --help) for a carefuler help')
		sys.exit(2)
	for opt, arg in opts:
		if opt in ("-h", "--help"):
			print('FlatClassifier.py -i <input_file> -n <levels_number> -t <level_target> (-f <features_number>|-p <packets_number>) -c <classifier_name>\n')
			print('Options:\n\t-i: dataset file, must be in arff format\n\t-n: number of levels (number of labels\' columns)\n\t-t: level target of classification, count of levels start from 1')
			print('\t-f or -p: former refers features number, latter refers packets number\n\t-c: classifier name choose from following list:')
			for sc in supported_classifiers:
				print('\t\t-c '+sc+'\t--->\t'+supported_classifiers[sc].split('_')[1]+'\t\timplemented in '+supported_classifiers[sc].split('_')[0])
			sys.exit()
		if opt in ("-i", "--input_file"):
			input_file = arg
		if opt in ("-n", "--levels_number"):
			levels_number = int(arg)
		if opt in ("-t", "--level_target"):
			level_target = int(arg)
		if opt in ("-f", "--nfeat"):
			features_number = int(arg)
		if opt in ("-p", "--npacket"):
			packets_number = int(arg)
		if opt in ("-c", "--clf"):
			classifier_name = arg

	if packets_number != 0 and features_number != 0 or packets_number == features_number:
		print('-f and -p option should not be used together')
		sys.exit()

	if levels_number == 0:
		print('Number of level must be positive and non zero')
		sys.exit()

	if level_target == 0 or level_target > levels_number:
		print('Level target must be positive, non zero and less than or equal to levels_number')
		sys.exit()

	if not input_file.endswith(".arff"):
		print('Input file must be .arff')
		sys.exit()

	if classifier_name not in supported_classifiers:
		print('Classifier not supported\nList of available classifiers:\n')
		for sc in supported_classifiers:
			print('-c '+sc+'\t--->\t'+supported_classifiers[sc].split('_')[1]+'\t\timplemented in '+supported_classifiers[sc].split('_')[0])
		sys.exit()

	flat_classifier = FlatClassifier(input_file=input_file,levels_number=levels_number,level_target=level_target,features_number=features_number,packets_number=packets_number,classifier_name=classifier_name)
	flat_classifier.kfold_validation(k=10)
