##ADAPTED FROM R SCRIPT (GDD-RF classifier)
#IMPORTS
import numpy as np
import pandas as pd
from pyfaidx import Fasta
import sys

def clinical_features(feature_table):
	#Gender
	feature_table = feature_table[feature_table.SEX.isin(['Male', 'Female'])]
	feature_table = feature_table.assign(Gender_F = feature_table.SEX.values == 'Female')
	feature_table = feature_table.assign(Gender_F = feature_table.Gender_F.astype(int))
	#MSI SCORE
	msi_inds = list(np.where(feature_table.MSI_SCORE==-1)[0])
	msi_inds.extend(np.where(np.isnan(feature_table.MSI_SCORE))[0])
	feature_table.loc[feature_table.index.isin(msi_inds),'MSI_SCORE'] = 0
	return feature_table

def purity_est(feature_table, maf_somatic, seg):
	#Classify low purity samples
	maf_somatic = maf_somatic.assign(t_depth = maf_somatic.t_alt_count + maf_somatic.t_ref_count)
	maf_somatic = maf_somatic.assign(t_var_freq = maf_somatic.t_alt_count / maf_somatic.t_depth)

	max_vaf = maf_somatic.groupby('SAMPLE_ID')['t_var_freq'].max()
	max_vaf = pd.DataFrame(max_vaf).reset_index()
	max_vaf.columns = ['SAMPLE_ID', 'max_vaf']

	max_logr = seg[seg['num.mark']>=100].groupby('SAMPLE_ID')['seg_absmean'].max()
	max_logr = pd.DataFrame(max_logr).reset_index()
	max_logr.columns = ['SAMPLE_ID', 'max_logr']

	purity_estimate = max_vaf.merge(max_logr, on = 'SAMPLE_ID', how = 'outer')
	feature_table = feature_table.merge(purity_estimate, on = 'SAMPLE_ID', how = 'left')
	feature_table.max_logr = feature_table['max_logr'].fillna(0)
	feature_table.max_vaf = feature_table['max_vaf'].fillna(0)
	#label low purity samples
	low_purity_inds = list(np.where((feature_table.max_vaf<0.1)&(feature_table.max_logr<0.2))[0])
	low_purity_inds.extend(list(np.where(feature_table.max_vaf<0)[0]))
	feature_table.loc[feature_table.index.isin(low_purity_inds),'Classification_Category'] = 'low_purity'
	feature_table = feature_table.drop(['max_vaf', 'max_logr'], axis =1)
	return feature_table

def tumor_mutational_burden(feature_table, maf_somatic, impact):
	#SNV COUNT, INDEL COUNT
	if impact == False:
		feature_table = feature_table.assign(LogSNV_Mb=0, LogINDEL_Mb=0)
	else:
		snv_count = pd.DataFrame(maf_somatic[maf_somatic.Variant_Type=='SNP'].SAMPLE_ID.value_counts()).reset_index()
		snv_count.columns = ['SAMPLE_ID', 'SNVCount']
		indel_count = pd.DataFrame(maf_somatic[maf_somatic.Variant_Type.isin(['INS', 'DEL'])].SAMPLE_ID.value_counts()).reset_index()
		indel_count.columns = ['SAMPLE_ID', 'INDELCount']
		mutation_count = snv_count.merge(indel_count, on = 'SAMPLE_ID', how = 'outer')
		feature_table = feature_table.merge(mutation_count, on = 'SAMPLE_ID', how = 'left')
		feature_table.SNVCount = feature_table['SNVCount'].fillna(0)
		feature_table.INDELCount = feature_table['INDELCount'].fillna(0)

		#ASSAY ADJUSTMENT, Log Mb by type calculations
		feature_table = feature_table.assign(SEQ_ASSAY_ID = [si[-3:] for si in feature_table.SAMPLE_ID])
		norm = 10**6
		canonical_capture_area = {'IM3':896665, 'IM5':1016478, 'IM6':1139322, 'IM7':1213770}
		feature_table.SNVCount = feature_table.SNVCount / [canonical_capture_area[panel]/norm if panel in canonical_capture_area.keys() else 'IM7' for panel in feature_table.SEQ_ASSAY_ID]
		feature_table.INDELCount = feature_table.INDELCount / [canonical_capture_area[panel]/norm if panel in canonical_capture_area.keys() else 'IM7' for panel in feature_table.SEQ_ASSAY_ID]
		feature_table = feature_table.assign(LogSNV_Mb = np.log10(feature_table.SNVCount+1))
		feature_table = feature_table.assign(LogINDEL_Mb = np.log10(feature_table.INDELCount+1))
		feature_table = feature_table.drop(['SNVCount', 'INDELCount', 'SEQ_ASSAY_ID'], axis = 1)
	return feature_table

def mutations(maf): 
	#only nonsyn consequences (mutations)
	maf['Consequence'] = maf.Consequence.fillna('')
	nonsyn_consequences = ["missense_variant", "stop_gained", "frameshift_variant", "splice_donor_variant", 
	"splice_acceptor_variant", "inframe_insertion", "inframe_deletion", "stop_lost", "exon_loss_variant",
	"disruptive_inframe_deletion", "disruptive_inframe_insertion", "start_lost"] #could maybe change these ? 
	nonsyn = ','.join(nonsyn_consequences)
	maf = maf.assign(rel_cons= [any(con in nonsyn for con in cons.split(',')) for cons in maf.Consequence])
	
	mut_d = maf[(maf.rel_cons==True)]
	
	#add TERTp 
	TERTp = maf[(maf.Hugo_Symbol=='TERT') & (maf.Consequence=='upstream_gene_variant')]
	TERTp = TERTp.assign(Hugo_Symbol = TERTp.Hugo_Symbol.replace({'TERT':'TERTp'}))
	mut_d = pd.concat([mut_d, TERTp])
	
	#aggregate across patients, make sure grabbing any version of the nonsyn consequence and using only 1 or 0
	mutations_d = pd.crosstab(index=mut_d['SAMPLE_ID'], columns=[mut_d['Hugo_Symbol']]).reset_index()
	mutations_d.iloc[:,1:] = np.where(mutations_d.iloc[:,1:]>=1,1,0)
	
	return mutations_d

def truncating_mutations(maf):    
	#only truncating mutations
	maf['Consequence'] = maf.Consequence.fillna('')
	trunc_consequences = ["stop_gained", "frameshift_variant", "splice_donor_variant",
			   "splice_acceptor_variant", "stop_lost", "exon_loss_variant"] 
	trunc = ','.join(trunc_consequences)
	maf = maf.assign(rel_cons_trunc= [any(con in trunc for con in cons.split(',')) for cons in maf.Consequence])
	   
	mut_trunc = maf[(maf.rel_cons_trunc==True)]
	
	#aggregate across patients, make sure grabbing any version of the nonsyn consequence and using only 1 or 0
	mutations_trunc = pd.crosstab(index=mut_trunc['SAMPLE_ID'], columns=[mut_trunc['Hugo_Symbol']]).reset_index()
	mutations_trunc.iloc[:,1:] = np.where(mutations_trunc.iloc[:,1:]>=1,1,0)
	return mutations_trunc

def focal_cna(cn):
	#aggregate by Gene
	cna = cn.groupby('Hugo_Symbol').sum()

	#amplification table
	cna_amp = pd.DataFrame(np.where(cna.iloc[:,:]>=2,1,0)).T
	cna_amp.columns = [hs + '_Amp' for hs in cna.index]
	cna_amp = cna_amp.assign(SAMPLE_ID=cna.columns)

	#deletion table
	cna_homdel = pd.DataFrame(np.where(cna.iloc[:,:]<=-2,1,0)).T
	cna_homdel.columns = [hs + '_HomDel' for hs in cna.index]
	cna_homdel = cna_homdel.assign(SAMPLE_ID=cna.columns)
	cna_data = pd.merge(cna_amp, cna_homdel, on = 'SAMPLE_ID')
	return cna_data

def broad_cna(seg, log_ratio_threshold):
	#load seg data
	seg.columns = ['SAMPLE_ID', 'chrom', 'start', 'end', 'num_mark', 'seg_mean', 'seg_absmean']
	seg = seg[(seg.num_mark>=10) & (seg.chrom!='Y')]
	seg = seg.assign(interval = [pd.Interval(start, end) for start, end in zip(seg.start, seg.end)])
	 
	#load reference ct table
	cytoband_table = pd.read_table('data/cytoband_table.txt')
	cytoband_table = cytoband_table[cytoband_table.chr!='Y']
	cytoband_table = cytoband_table.assign(length = (cytoband_table.end - cytoband_table.start) + 1)
	
	#iterate over each arm and add to seg_data
	seg_data = None
	for arm in cytoband_table.arm.values:
		sub_seg = seg[seg.chrom==arm[:-1]]
		arm_start, arm_end = cytoband_table[cytoband_table.arm==arm][['start', 'end']].values[0]
		arm_int = pd.Interval(arm_start, arm_end)
		arm_len = (arm_end - arm_start) + 1
		sub_seg = sub_seg.assign(overlap = [interval.overlaps(arm_int) for interval in sub_seg.interval])
		sub_seg = sub_seg.assign(width = [(min(arm_end, seg_end) - max(arm_start, seg_start))+1 for seg_start, seg_end in zip(sub_seg.start, sub_seg.end)])
		#deletions
		sub_del = sub_seg[(sub_seg.overlap) & (sub_seg.seg_mean <= -log_ratio_threshold)]
		del_coverage = pd.DataFrame(sub_del.groupby('SAMPLE_ID')['width'].sum()/arm_len)
		del_coverage.columns = ['coverage']
		del_coverage = del_coverage.assign(CNA = 'Del_' + arm)
		del_coverage = del_coverage[del_coverage.coverage>=.5].reset_index()
		#amplifications
		sub_amp = sub_seg[(sub_seg.overlap) & (sub_seg.seg_mean >= log_ratio_threshold)]
		amp_coverage = pd.DataFrame(sub_amp.groupby('SAMPLE_ID')['width'].sum()/arm_len)
		amp_coverage.columns = ['coverage']
		amp_coverage = amp_coverage.assign(CNA = 'Amp_' + arm)
		amp_coverage = amp_coverage[amp_coverage.coverage>=.5].reset_index()
		
		#concat
		arm_cnas = pd.concat([del_coverage, amp_coverage], axis = 0)
		if seg_data is None:
			seg_data = arm_cnas
		else:
			seg_data = pd.concat([seg_data, arm_cnas])

	#aggregate across patients
	seg_data = pd.crosstab(seg_data.SAMPLE_ID, seg_data.CNA).reset_index()
	seg = seg.assign(width = (seg.end - seg.start)+1)
	tot_cov = pd.DataFrame(seg.groupby('SAMPLE_ID')['width'].sum()).reset_index()
	rel_cov = pd.DataFrame(seg[seg.seg_absmean>=log_ratio_threshold].groupby('SAMPLE_ID')['width'].sum()).reset_index()

	#calculate copy number burden
	cn_burden_dt = pd.merge(tot_cov, rel_cov, on = 'SAMPLE_ID', suffixes = ['_total', '_relevant'])
	cn_burden_dt = cn_burden_dt.assign(CN_Burden = 100*np.round(cn_burden_dt.width_relevant/cn_burden_dt.width_total, 3))
	seg_data = pd.merge(seg_data, cn_burden_dt, on = 'SAMPLE_ID', how = 'right')
	seg_data = seg_data.drop(['width_total', 'width_relevant'], axis = 1)
	seg_data = seg_data.fillna(0)
	return seg_data

def fusions(SV):
	#grab fusions, intragenic fusions
	fusion_list = pd.read_csv('data/fusions.txt', sep = '\t')
	fusion_genes = set(fusion_list[~(fusion_list.Gene_Fusion.isna())].Gene_Fusion)
	fusion_intragenic = set(fusion_list[~(fusion_list.Intragenic_Fusion.isna())].Intragenic_Fusion) 
	SV_data = SV[(SV.Hugo_Symbol.isin(fusion_genes)) | (SV.Fusion.isin(fusion_intragenic))] #each fusion is listed twice per sample with each fusion partner
	fusion_dic = {}
	for gene in fusion_genes:
		fusion_dic[gene] = gene + '_fusion'
	for fusion in fusion_intragenic:
		gene = fusion.split('-intragenic')[0]
		fusion_dic[gene] = gene + '_SV'
	SV_data = SV_data.assign(fusion_label = SV_data.Hugo_Symbol.replace(fusion_dic))
	#aggregate by patient
	SV_data = pd.crosstab(index = SV_data.SAMPLE_ID, columns = SV_data.fusion_label)
	SV_data.iloc[:,1:] = np.where(SV_data.iloc[:,1:]>=1,1,0)
	return SV_data

def hotspots(maf):
	#load hotspot list
	hotspot_list = pd.read_csv('data/final_hotspot_list.csv', index_col = 0)
	maf = maf.assign(Hotspot_Label = [str(hs) + str(hgsvp) for hs, hgsvp in zip(maf.Hugo_Symbol, maf.HGVSp_Short)])
	maf['Consequence'] = maf.Consequence.fillna('')
	nonsyn_consequences = ["missense_variant", "stop_gained", "frameshift_variant", "splice_donor_variant", 
	"splice_acceptor_variant", "inframe_insertion", "inframe_deletion", "stop_lost", "exon_loss_variant",
	"disruptive_inframe_deletion", "disruptive_inframe_insertion", "start_lost"] #could maybe change these ? 
	nonsyn = ','.join(nonsyn_consequences)
	maf = maf.assign(rel_cons= [any(con in nonsyn for con in cons.split(',')) for cons in maf.Consequence])
	mut_hs = maf[(maf.rel_cons==True)&(maf.Hotspot_Label.isin(hotspot_list.Hotspot_Label))]
	
	#aggregate across patients - allele level, gene level
	hotspots_d = pd.crosstab(index=mut_hs['SAMPLE_ID'], columns=[mut_hs['Hugo_Symbol']]).reset_index()
	hotspots_d.columns = ['SAMPLE_ID'] + [i + '_hotspot' for i in hotspots_d.columns[1:]]
	hotspots_d_allele = pd.crosstab(index=mut_hs['SAMPLE_ID'], columns=[mut_hs['Hotspot_Label']]).reset_index()
	hotspots_d = hotspots_d.merge(hotspots_d_allele, on = 'SAMPLE_ID')
	hotspots_d.iloc[:,1:] = np.where(hotspots_d.iloc[:,1:]>=1,1,0)
	hotspots_d.columns = [col.replace('p.', '.') for col in hotspots_d]
	return hotspots_d

def sbs_counts(maf_all):
	#load fasta
	ref_fasta = Fasta(path_to_fasta) #will error out if specified incorrectly

	maf = maf_all[(maf_all.Variant_Type== 'SNP')]
	maf = maf.assign(ident = [chrom + '_' + str(start-2) + '_' + str(end+1) for chrom, start, end in zip(maf.Chromosome, maf.Start_Position, maf.End_Position)])
	
	#classify substitutions from unknown, germline SNP mutations in the maf
	substitution_order = ["ACAA", "ACAC", "ACAG", "ACAT", "CCAA", "CCAC", "CCAG", "CCAT", "GCAA", "GCAC", "GCAG", "GCAT", "TCAA", "TCAC", "TCAG", "TCAT", "ACGA", "ACGC", "ACGG", "ACGT", "CCGA", "CCGC", "CCGG", "CCGT", "GCGA", "GCGC", "GCGG", "GCGT", "TCGA", "TCGC", "TCGG", "TCGT", "ACTA", "ACTC", "ACTG", "ACTT", "CCTA", "CCTC", "CCTG", "CCTT", "GCTA", "GCTC", "GCTG", "GCTT", "TCTA", "TCTC", "TCTG", "TCTT", "ATAA", "ATAC", "ATAG", "ATAT", "CTAA", "CTAC", "CTAG", "CTAT", "GTAA", "GTAC", "GTAG", "GTAT", "TTAA", "TTAC", "TTAG", "TTAT", "ATCA", "ATCC", "ATCG", "ATCT", "CTCA", "CTCC", "CTCG", "CTCT", "GTCA", "GTCC", "GTCG", "GTCT", "TTCA", "TTCC", "TTCG", "TTCT", "ATGA", "ATGC", "ATGG", "ATGT", "CTGA", "CTGC", "CTGG", "CTGT", "GTGA", "GTGC", "GTGG", "GTGT", "TTGA", "TTGC", "TTGG", "TTGT"]
	pyrimidines = ['C', 'T']
	purines = ['G', 'A']
	swap_dic = {'A':'T','T':'A','C':'G','G':'C'}
	fasta_dic = {}
	def norm(tnc):
		#normalize so that pyrimidines in the center
		if tnc[1] not in pyrimidines:
			return ''.join([swap_dic[nt] for nt in tnc[::-1]])
		return tnc
	
	for ident in set(maf.ident):
		chrom, start, end = ident.split('_')
		try:
			chrom, start, end = chrom, int(start), int(end)
			fasta_dic[ident] = norm(ref_fasta[chrom][start:end].seq.upper())

		except:
			chrom, start, end = 'chr' + chrom, int(start), int(end)
			fasta_dic[ident] = norm(ref_fasta[chrom][start:end].seq.upper())
		
	maf = maf.assign(ref_tri = [fasta_dic[ident] for ident in maf.ident])
	maf = maf.assign(norm_alt = [alt if ref in pyrimidines else swap_dic[alt] for alt, ref in zip(maf.Tumor_Seq_Allele2, maf.Reference_Allele)])
	maf = maf.assign(transition_form = [ref_tri[:2] + alt + ref_tri[-1] for ref_tri, alt in zip(maf.ref_tri, maf.norm_alt)])
	#aggregate across patients
	sigs_data_raw = pd.crosstab(index=maf['SAMPLE_ID'], columns=maf['transition_form']).reset_index()
	return sigs_data_raw

def signatures(sigs):
	#relevant signatures
	sigs = sigs.drop([col for col in sigs.columns if 'confidence' in col], axis = 1)
	sigsm = pd.melt(sigs, id_vars = ("SAMPLE_ID", "Nmut", "impact_version", "Nmut_Mb"), var_name = ('Signature'), value_name = 'exposure')
	sigsm.Signature = [sig.replace('mean_', '') for sig in sigsm.Signature]
	sig_names = {"1":"Sig_Age",
			 "2":"Sig_APOBEC",
			 "3":"Sig_BRCA",
			 "4":"Sig_Smoking",
			 "5":"Other",
			 "6":"Sig_MMR",
			 "7":"Sig_UV",
			 "8":"Other",
			 "9":"Other",
			 "10":"Sig_POLE",
			 "11":"Sig_TMZ",
			 "12":"Other",
			 "13":"Sig_APOBEC",
			 "14":"Other",
			 "15":"Sig_MMR",
			 "16":"Other",
			 "17":"Other",
			 "18":"Other",
			 "19":"Other",
			 "20":"Sig_MMR",
			 "21":"Other",
			 "22":"Other",
			 "23":"Other",
			 "24":"Sig_Smoking",
			 "25":"Other",
			 "26":"Sig_MMR",
			 "27":"Other",
			 "28":"Other",
			 "29":"Other",
			 "30":"Other"}

	sigsm = sigsm.assign(sig_name = sigsm.Signature.replace(sig_names))
	sigsm = sigsm[(sigsm.Nmut>10)&(sigsm.sig_name!='Other')] #only want relevant signatures, when NMut is greater than 10, exposure greater than .4 overall
	sigsm = pd.DataFrame({'exposure' : sigsm.groupby(['SAMPLE_ID', 'sig_name'])['exposure'].sum()}).reset_index()
	sigsm = sigsm[sigsm.exposure>=.4]
	#aggregate across patients
	sig_data = pd.crosstab(index=sigsm['SAMPLE_ID'], columns=sigsm['sig_name']).reset_index()
	return sig_data

#parse args 
#format : python generate_ft_table.py path/to/fasta path/to/ft
#both fasta and output ft filename are required

if len(sys.argv) < 3:
	raise Exception("Not enough arguments")

path_to_fasta = str(sys.argv[1])
path_to_ft = str(sys.argv[2])

print('Specified Fasta:', path_to_fasta)
print('Output Table:', path_to_ft)

#set file path
impact = True
#load clinical data
data_clinical_sample = pd.read_table('data/msk_solid_heme/data_clinical_sample.txt', skiprows = 4)
data_clinical_patient = pd.read_table('data/msk_solid_heme/data_clinical_patient.txt', skiprows = 4)
data_clinical_sample = pd.merge(data_clinical_sample, data_clinical_patient)
data_clinical_sample = data_clinical_sample[data_clinical_sample.SAMPLE_ID.str.contains('IM')] #only looking at MSK-IMPACT patients

#load cancertypes
input_cancertypes = pd.read_table('data/tumor_type_final.txt') #need to fix
feature_table = pd.merge(data_clinical_sample, input_cancertypes, how = 'left', on = ['CANCER_TYPE', 'CANCER_TYPE_DETAILED'])
feature_table.Cancer_Type = feature_table.Cancer_Type.fillna('other')
feature_table = feature_table.assign(Classification_Category = ['train' if ct != 'other' else 'other' for ct in feature_table.Cancer_Type])

#Signatures
sigs = pd.read_table('data/msk_solid_heme/msk_solid_heme_data_mutations_unfiltered.sigs.tab.txt')
sigs = sigs.rename({'Tumor_Sample_Barcode':'SAMPLE_ID'}, axis = 1)

#CN Data
seg = pd.read_table('data/msk_solid_heme/mskimpact_data_cna_hg19.seg')
seg = seg.assign(seg_absmean=abs(seg['seg.mean']))
seg = seg.rename({'ID':'SAMPLE_ID'}, axis = 1)
cn = pd.read_table('data/msk_solid_heme/data_CNA.txt')

#Fusions
SV = pd.read_table('data/msk_solid_heme/data_fusions.txt')
SV = SV.rename({'Tumor_Sample_Barcode':'SAMPLE_ID', 'Sample_ID':'SAMPLE_ID','Site2_Contig':'Hugo_Symbol'}, axis = 1)

#Mutations 
maf = pd.read_table('data/msk_solid_heme/data_mutations_extended.txt', skiprows = 1)
maf = maf.rename({'Tumor_Sample_Barcode':'SAMPLE_ID'}, axis =1)
maf = maf.sort_values(by = 'SAMPLE_ID')
maf_all_somatic = maf[maf.Mutation_Status=='SOMATIC'] #calculate purity estimates, TMB from all detected somatic mutations, not just impact-341 

#correct Hugo Symbols for current notation, only include msk_impact 341 genes
impact_genes = pd.read_excel('data/IMPACT505_Gene_list_detailed.xlsx', sheet_name = '505 genes')
impact_genes.columns = ['Hugo_Symbol', 'Name', 'N_exons', 'Panel']
alt_names_list = pd.read_excel('data/IMPACT505_Gene_list_detailed.xlsx', sheet_name = 'alt_names')
alt_names = {}
for hs_old, hs_current in zip(alt_names_list.HS_OLD.values, alt_names_list.HS_CURRENT.values):
	alt_names[hs_old] = hs_current
msk_impact_341 = impact_genes[impact_genes.Panel=='IMPACT-341'].Hugo_Symbol

maf.Hugo_Symbol = maf.Hugo_Symbol.replace(alt_names)
cn.Hugo_Symbol = cn.Hugo_Symbol.replace(alt_names)

#only include MSK IMPACT 341 genes for other calculations
maf = maf[maf.Hugo_Symbol.isin(msk_impact_341)]
cn = cn[cn.Hugo_Symbol.isin(msk_impact_341)]

print('loads completed')
#separate maf into somatic vs unknown & somatic
maf_all = maf[maf.Mutation_Status!='GERMLINE'] #only not-germline
maf_somatic = maf[maf.Mutation_Status=='SOMATIC'] #only somatic

#extract clinical features (gender, MSI Score)
feature_table = clinical_features(feature_table)
feature_table = feature_table[['SAMPLE_ID', 'CANCER_TYPE', 'CANCER_TYPE_DETAILED', 'Cancer_Type', 'SAMPLE_TYPE', 'PRIMARY_SITE', 'METASTATIC_SITE', 'Classification_Category', 'Gender_F', 'MSI_SCORE']]
#remove low purity
feature_table = purity_est(feature_table, maf_all_somatic, seg)
print('clinical integrated')
#mutations
feature_table = tumor_mutational_burden(feature_table, maf_all_somatic, impact)
feature_table = feature_table.merge(mutations(maf_somatic), on = 'SAMPLE_ID', how = 'left').fillna(0)
feature_table = feature_table.merge(truncating_mutations(maf_somatic), on = 'SAMPLE_ID', how = 'left', suffixes = [None, '_TRUNC']).fillna(0)
print('mutations integrated')
#hotspots
feature_table = feature_table.merge(hotspots(maf_somatic), on = 'SAMPLE_ID', how = 'left').fillna(0)
print('hotspots integrated')
#cn
feature_table = feature_table.merge(focal_cna(cn), on = 'SAMPLE_ID', how = 'left').fillna(0)
feature_table = feature_table.merge(broad_cna(seg, log_ratio_threshold = .2), on = 'SAMPLE_ID', how = 'left').fillna(0)
print('copy number integrated')
#fusions
feature_table = feature_table.merge(fusions(SV), on = 'SAMPLE_ID', how = 'left').fillna(0)
print('fusions integrated')
#signatures
feature_table = feature_table.merge(signatures(sigs), on = 'SAMPLE_ID', how = 'left').fillna(0)
feature_table = feature_table.merge(sbs_counts(maf_all), on = 'SAMPLE_ID', how = 'left').fillna(0)
print('signatures integrated')

#save feature table
feature_table.to_csv(path_to_ft)

