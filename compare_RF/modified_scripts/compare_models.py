import pandas as pd
import sklearn.metrics

def metrics(true, pred, rep = False):
    accuracy = sklearn.metrics.accuracy_score(true, pred)
    precision = sklearn.metrics.precision_score(true, pred, average = 'macro')
    recall = sklearn.metrics.recall_score(true, pred, average = 'macro')
    f1 = sklearn.metrics.f1_score(true, pred, average = 'macro')
    if rep:
        report = sklearn.metrics.classification_report(true, pred, output_dict = True)
        return accuracy, precision, recall, f1, report

    else:
        return accuracy, precision, recall, f1
    
def data_form_lbl(res, true_col = 'true', pred_col = 'pred', probs_col = 'probs'):
    acc, mp = metrics(res[true_col], res[pred_col])[0:2]
    hc_acc, hc_mp = metrics(res[res[probs_col]>=.75][true_col], res[res[probs_col]>=.75][pred_col])[0:2]
    df = pd.DataFrame([acc, mp, len(res[res[probs_col]>=.75])/len(res), hc_acc, hc_mp]).T
    df.columns = ['Acc.', 'Macro-Prec.', 'Prop. High-Conf.', 'High-Conf. Acc.', 'High-Conf. Macro-Prec.']
    return df

orig_rf_res = pd.read_excel('preloaded_results/rf_results_orig.xlsx', sheet_name = 1)
ctypes_22 = sorted(list(orig_rf_res.columns[10:32]))
orig_ens_res = pd.read_csv('preloaded_results/ensemble_results_orig.csv')
orig_ens_res = orig_ens_res.assign(true_label = orig_rf_res.Cancer_Type.values, pred_label = [ctypes_22[i] for i in orig_ens_res.pred])

orig_rf_row = data_form_lbl(orig_rf_res, 'Cancer_Type', 'Pred1', 'Conf1')
orig_ens_row = data_form_lbl(orig_ens_res, 'true_label', 'pred_label', 'probs')
orig_rf_row = orig_rf_row.assign(Model = 'RF', Test_Set = 'GDD RF EXT')
orig_ens_row = orig_ens_row.assign(Model = 'ENS', Test_Set = 'GDD RF EXT')


upd_rf_res = pd.read_excel('preloaded_results/rf_results_upd.xlsx')
ctypes_38 = sorted(set(upd_rf_res.Cancer_Type))
upd_ens_res = pd.read_csv('preloaded_results/ensemble_results_upd.csv')
upd_ens_res = upd_ens_res.assign(true_label = [ctypes_38[i] for i in upd_ens_res.true], pred_label = [ctypes_38[i] for i in upd_ens_res.pred])

upd_rf_row = data_form_lbl(upd_rf_res, 'Cancer_Type', 'Pred1', 'Conf1')
upd_ens_row = data_form_lbl(upd_ens_res, 'true_label', 'pred_label', 'probs')
upd_rf_row = upd_rf_row.assign(Model = 'RF', Test_Set = 'GDD ENS')
upd_ens_row = upd_ens_row.assign(Model = 'ENS', Test_Set = 'GDD ENS')
final_comp_res = pd.concat([orig_rf_row, orig_ens_row, upd_rf_row, upd_ens_row])[['Model', 'Test_Set', 'Acc.', 'Macro-Prec.', 'Prop. High-Conf.', 'High-Conf. Acc.', 'High-Conf. Macro-Prec.']]
print(final_comp_res)
final_comp_res.to_csv('final_comp_res.csv')
