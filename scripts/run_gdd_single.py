import sys, os, shap, re, json
import numpy as np 
import pandas as pd 

import joblib

import torch 
import torch.nn as nn 
import torch.nn.functional as F 


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class MLP(torch.nn.Module):
	#multi-layer perceptron class
	def __init__(self, num_fc_layers, num_fc_units,  dropout_rate, n_features, n_types):
		super().__init__()
		#fc_layers = number of layers that the MLP will have
		#fc_units = number of units in each of the middle layers
		self.layers = nn.ModuleList() #empty module list as of rn
		self.layers.append(nn.Linear(n_features, num_fc_units)) #(in features, out_features)
		for i in range(num_fc_layers):
			self.layers.append(nn.Linear(num_fc_units, num_fc_units)) #linear unit
			self.layers.append(nn.ReLU(True)) #reLu activation
			self.layers.append(nn.Dropout(p=dropout_rate))
		self.layers.append(nn.Linear(num_fc_units, n_types)) #in features, out_features (hard coded)
	#functions which run the model
	def forward(self, x): 
		for i in range(len(self.layers)): 
			x = self.layers[i](x)

		return x

	def feature_list(self, x): 
		out_list = []
		for i in range(len(self.layers)): 
			x = self.layers[i](x)
			out_list.append(x)
		return out_list

	def intermediate_forward(self, x, layer_index):
		for i in range(layer_index): 
			x = self.layers[i](x)
		return x


class EnsembleClassifier(nn.Module): 
	#Ensemble Class
	def __init__(self, model_list): 
		super(EnsembleClassifier, self).__init__()
		self.model_list = model_list

	def forward(self, x): 
		logit_list = [] 
		for model in self.model_list: 
			model.eval()
			logits = model(x)
			logit_list.append(logits)
		return logit_list 

def process_data_single(single_data, colnames): 
	### generate a single feature table to run through GDD-ENS 
	if 'CANCER_TYPE' not in single_data.columns:
		single_data = single_data.assign(CANCER_TYPE='NA')
	if 'Classification_Category' not in single_data.columns:
		single_data = single_data.assign(Classification_Category='NA')
	values = [single_data[col].values[0] if col in single_data.columns else 0 for col in colnames]
	values = pd.DataFrame(values).T
	values.columns = colnames
	return single_data, values

def pred_results(model, data, ctypes):
	#similar to softmax_predictive_accuracy function
	fold_logits = model(data)
	probs_list = [F.softmax(logits, dim=1) for logits in fold_logits]
	fold_preds = [torch.max(probs_list[i], 1)[1].cpu().data.numpy() for i in range(len(probs_list))]
	probs_tensor = torch.stack(probs_list, dim = 2)
	probs = torch.mean(probs_tensor, dim=2)
	allprobs = probs.cpu().data.numpy()
	top_probs = sorted(allprobs[0], reverse = True)[:3]
	top_preds = sorted(range(len(allprobs[0])), key = lambda x: allprobs[0][x], reverse = True)[:3]
	pred_labels = [ctypes[int(pred)] for pred in top_preds]
	return top_preds, top_probs, pred_labels, allprobs


def top_shap_single(model, test_x, test_y, colnames):
	#validate shapley values by calculating across all types
	precalc_shap = joblib.load(filename='data/gddnn_kmeans_output.bz2')
	feature_annotation = pd.read_csv('data/feature_annotations.csv', index_col = 0)

	pred_ind = test_y
	def spec_prob_function(x):
			#use this to grab probability from ensemble classifier (SHAP package formatting)
		x = torch.from_numpy(x).float()
		x = x.to(device)
		logits_list = model(x)
		probs_list = [F.softmax(logits, dim=1) for logits in logits_list]
		probs_tensor = torch.stack(probs_list, dim = 2)
		probs = torch.mean(probs_tensor, dim=2)
		pred_probs= probs[:,pred_ind]
		pred_probs = pred_probs.cpu().data.numpy().reshape(len(x),)
		return pred_probs
	explainer = shap.KernelExplainer(spec_prob_function, precalc_shap)
	shap_values = explainer.shap_values(test_x, nsamples=2000)
		
	#format data
	shap_values = pd.DataFrame(shap_values)
	shap_values = shap_values.append(pd.DataFrame(test_x))
	shap_values.columns = colnames
	shap_values.index = ['Shapley_Values', 'Feature_Values']
	shap_values = shap_values[[i for i in shap_values.columns if i in feature_annotation.in_col.values]]
	shap_values = shap_values.T
	shap_values.sort_values(by = 'Shapley_Values', axis = 0, inplace=True, ascending=False)
	shap_cols = []
	for col, fv in zip(shap_values.index, shap_values.Feature_Values):
		new_col = feature_annotation[(feature_annotation.in_col==col)]
		if len(new_col)==2:
			shap_cols.append(new_col[new_col.value==fv].out_col.values[0])
		else:
			shap_cols.append(new_col.out_col.values[0])
	shap_values = shap_values.assign(Shapley_Columns = shap_cols)
	return shap_values[:10] 

def format_gdd_output(single_data, res, allprobs, shap):
	def str_format(raw_list, round_int = 4):
		form_list = ["{:.2e}".format(i) if str(round(i, round_int))[:4] == '0.00' else str(round(i, round_int)) for i in raw_list]
		return form_list
	prob_form = str_format(res.prob.values)
	shap_form = str_format(shap.Shapley_Values.values)
	allprobs_form = str_format(list(allprobs[0]), round_int = 6)
	final_cols = ['SAMPLE_ID', 'Cancer_Type', 'Classification_Category', 'Pred1', 'Conf1', 'Pred2', 'Conf2', 'Pred3', 'Conf3'] + ctypes
	final_res = [single_data.SAMPLE_ID.values[0], single_data.CANCER_TYPE.values[0], single_data.Classification_Category.values[0], res.pred_label.values[0], prob_form[0], res.pred_label.values[1], prob_form[1], res.pred_label.values[2], prob_form[2]] + allprobs_form

	final_cols.extend(['Var1', 'Imp1', 'Var2', 'Imp2',  'Var3', 'Imp3',  'Var4', 'Imp4', 'Var5', 'Imp5',  'Var6', 'Imp6',  'Var7', 'Imp7',  'Var8', 'Imp8',  'Var9', 'Imp9', 'Var10', 'Imp10'])
	sv_res = []
	for fv, imp in zip(shap.Shapley_Columns, shap_form):
		sv_res.extend([fv, imp])
	final_res.extend(sv_res)
	final_res = pd.DataFrame(final_res).T
	final_res.columns = final_cols	
	return final_res

if __name__ == "__main__":
	torch.manual_seed(3407)
	np.random.seed(0)
	
	sys.path.insert(0, '../')

	if len(sys.argv) < 3:
		raise Exception("Not enough arguments")

	#load data, column names, cancer types and annotations
	model_path = sys.argv[1]
	predict_single_fn = sys.argv[2]
	single_data = pd.read_csv(predict_single_fn)
	colnames = pd.read_csv('data/ft_colnames.csv')['columns'].values
	ctypes = list(pd.read_csv('data/tumor_type_ordered.csv')['Cancer_Type'].values)

	#process single instance, save
	single_data, gdd_input = process_data_single(single_data, colnames)
	#run predictions
	gdd_data = torch.from_numpy(np.array(gdd_input)).float()
	gdd_data = gdd_data.to(device)
	# gdd_model = torch.load('/data/bergerm1/ch_GDDP2/ensemble.pt')
	gdd_model = torch.load(model_path, map_location=torch.device(device))
	print('model loaded:', model_path)

	preds, probs, pred_label, allprobs = pred_results(gdd_model, gdd_data, ctypes) 
	res = pd.DataFrame([preds,probs,pred_label]).T
	res.columns = ['pred', 'prob', 'pred_label']

	#calculate shapley values
	gdd_x = np.array(gdd_input)
	gdd_y = np.array([preds[0]])
	shap_values = top_shap_single(gdd_model, gdd_x, gdd_y, colnames)
	if len(shap_values.index) >= 10:
		top_shap = shap_values[:10]
	else:
		fill = pd.DataFrame([['NA', "NA", 0] for i in range(10-len(shap_values))])
		fill.columns = ['Feature_Values', 'Shapley_Columns', 'Shapley_Values']
		top_shap = pd.concat([shap_values, fill])

	#format final res
	final_res = format_gdd_output(single_data, res, allprobs, top_shap)
	final_res_fn = sys.argv[3]
	final_res.to_csv(final_res_fn)
