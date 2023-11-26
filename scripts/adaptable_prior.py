import numpy as np
import pandas as pd 
import sklearn.metrics
import sklearn.calibration
import sys

def metrics(true, pred, rep = False):
    accuracy = sklearn.metrics.accuracy_score(true, pred)
    precision = sklearn.metrics.precision_score(true, pred, average = 'macro')
    recall = sklearn.metrics.recall_score(true, pred, average = 'macro')
    f1 = sklearn.metrics.f1_score(true, pred, average = 'macro')
    #report = sklearn.metrics.classification_report(true, pred, output_dict = True)
    if rep:
        report = sklearn.metrics.classification_report(true, pred, output_dict = True)
        return accuracy, precision, recall, f1, report

    else:
        return accuracy, precision, recall, f1

def make_prior_dics(prior_table, type_labels):
    #load training for original dist and all priors provided in prior_table
    curated_dic_list = {}
    for col in prior_table.columns:
        curated_dic = {}
        for i in range(len(prior_table.index)):
            curated_dic[type_labels[i]] = prior_table[prior_table.index==type_labels[i]][col].values[0]
        curated_dic_list[col] = curated_dic

    return curated_dic_list


def prior_adjust(prior_dic_list, output, allprobs, type_labels, new_output_file):
    #adjust given the biopsy site for a pre-defined site, for as many examples as are in output and as many priors are provided in prior_table
    new_preds = []
    new_probs = []
    for i in range(len(output.index)):
        prob_array = allprobs.iloc[i]
        prevalence_training = [prior_dic_list['original'][ct] for ct in type_labels]
        prev_desired_all = []
        for prior in prior_dic_list.keys():
            if prior != 'original':
                prevalence_desired = [prior_dic_list[prior][ct] for ct in type_labels]
                prev_desired_all.append(prevalence_desired)
        rescale = []
        for p_t, t in zip(prob_array, range(len(type_labels))):
            ct_des = [prev[t] for prev in prev_desired_all]
            ct_prior = prevalence_training[t] ** len(ct_des)
            ct_des = np.prod(ct_des)
            p_adj = p_t*ct_des / ct_prior
            rescale.append(p_adj)
        rescale_array = rescale/ sum(rescale)
        #print('rescale', rescale_array)
        new_preds.append(np.argmax(rescale_array))
        new_probs.append(np.max(rescale_array))
        output_file_label = new_output_file.split('.')[0]
        pd.DataFrame(rescale_array).to_csv(output_file_label + '_rescale_array.csv')
    output = output.assign(orig_pred = output.Pred1)
    output = output.assign(orig_prob = output.Conf1)
    output = output.assign(new_pred= [type_labels[pred] for pred in new_preds])
    output = output.assign(new_prob= new_probs)
    output.to_csv(new_output_file) 

    return output



if __name__ == "__main__":
    #load datasets 
    if len(sys.argv) < 4:
        raise Exception("Not enough arguments")

    prior_file, output_file, new_output_file = sys.argv[1], sys.argv[2], sys.argv[3]
    sys.path.insert(0, '../')

    prior_table = pd.read_csv(prior_file, index_col = 0)
    type_labels = sorted(list(prior_table.index.values))
    output_test = pd.read_csv(output_file, index_col = 0)
    allprobs = output_test[[i for i in output_test.columns if i in type_labels]]
    prior_dic_list = make_prior_dics(prior_table, type_labels)
    output_test = prior_adjust(prior_dic_list, output_test, allprobs, type_labels, new_output_file)
