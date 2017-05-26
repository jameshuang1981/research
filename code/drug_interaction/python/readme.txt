
This is the instruction for running the python code to discover causal relationships from time series data.



Please cite the following paper when using the code



Here is the command for generating the source setting from terminal 
python3 generate_setting.py /Users/yuxiaohuang/Documents/github/research/code/drug_interaction/simulated_data/setting/source/src_setting_0.txt 10 20 0.5 1.0 'src'



Here is the command for generating the target setting from terminal 
python3 generate_setting.py /Users/yuxiaohuang/Documents/github/research/code/drug_interaction/simulated_data/setting/target/trg_setting_0.txt 1 1 0 0.1 'trg'



Here is the command for generating the causal pie from terminal 
python3 generate_causal_pie.py /Users/yuxiaohuang/Documents/github/research/code/drug_interaction/simulated_data/setting/source/src_setting_0.txt /Users/yuxiaohuang/Documents/github/research/code/drug_interaction/simulated_data/setting/target/trg_setting_0.txt /Users/yuxiaohuang/Documents/github/research/code/drug_interaction/simulated_data/causal_pie/causal_pie_0.txt 1 3 1 3 1 5 0.5 1.0



Here is the command for generating the data from terminal 
python3 generate_data.py /Users/yuxiaohuang/Documents/github/research/code/drug_interaction/simulated_data/setting/source/src_setting_0.txt /Users/yuxiaohuang/Documents/github/research/code/drug_interaction/simulated_data/setting/target/trg_setting_0.txt /Users/yuxiaohuang/Documents/github/research/code/drug_interaction/simulated_data/causal_pie/causal_pie_0.txt /Users/yuxiaohuang/Documents/github/research/code/drug_interaction/simulated_data/data/source/src_0.txt /Users/yuxiaohuang/Documents/github/research/code/drug_interaction/simulated_data/data/target/trg_0.txt 3600



Here is the command for running the code from terminal 
python3 search_4_causal_pie.py /Users/yuxiaohuang/Documents/github/research/code/drug_interaction/simulated_data/data/source/src_0.txt /Users/yuxiaohuang/Documents/github/research/code/drug_interaction/simulated_data/data/target/trg_0.txt /Users/yuxiaohuang/Documents/github/research/code/drug_interaction/simulated_data/result/pie.txt /Users/yuxiaohuang/Documents/github/research/code/drug_interaction/simulated_data/log/log.txt /Users/yuxiaohuang/Documents/github/research/code/drug_interaction/simulated_data/figure/ 0 0.01 30 1 5



It looks like this when I tested the code on the toy dataset in the test folder         
python alpha.py /Users/yuxiaohuang/Documents/github/research/code/FLAIRS_15/python/test/data/0_disc.csv /Users/yuxiaohuang/Documents/github/research/code/FLAIRS_15/python/test/data/0_cont.csv True True all /Users/yuxiaohuang/Documents/github/research/code/FLAIRS_15/python/test/result/0_alpha.csv /Users/yuxiaohuang/Documents/github/research/code/FLAIRS_15/python/test/result/0_sig_rel.csv all 0.01 both 1 1



Here disc_data_file and cont_data_file are the input files, while alpha_file and sig_rel_file are the output files. Specifically, the meaning of the parameters are as follows
alpha.py        :   the python code

disc_data_file  :   discrete data file, which includes variables that can be the causes, the data are of the following form
		     var1 (i.e. header), var1_t1 (i.e. val), ..., var1_tn (i.e. val)
		     , ...,
		     varn (i.e. header), varn_t1 (i.e. val), ..., varn_tn (i.e. val)

cont_data_file  :   continuous data file, which includes variables that can be the effects, the data are of the following form
		     var1 (i.e. header), var1_t1 (i.e. val), ..., var1_tn (i.e. val)
		     , ...,
		     varn (i.e. header), varn_t1 (i.e. val), ..., varn_tn (i.e. val)

header          :   tells us whether the data files have a header
		     True,  if there is a header
		     False, otherwise

transpose       :   tells us whether the data need to be transposed
                    False, when the data are of the above form, thus do not need to be transposed
                    True,  when the data are of the following form, thus need to be transposed
                    var1 (i.e. header), ..., varn (i.e. header)
		     var1_t1 (i.e. val), ..., varn_t1 (i.e. val)
		     , ...,
		     var1_tn (i.e. val), ..., varn_tn (i.e. val)

rel_type        :   tells us the type of hypotheses we want to test
                    "not_equal" for hypotheses s.t. E [e | c] != E[e]
                    "positive"  for hypotheses s.t. E [e | c] > E[e]
                    "negative"  for hypotheses s.t. E [e | c] < E[e]
                    "all"       for all hypotheses

alpha_file      :   file containing relationships and their causal significance (i.e. alpha)

sig_rel_file    :   file containing significant relationships

family_type     :   tells us whether to test the relationships for all effects (all families) at a time, or the relationships for each effect (one family) at a time
		     "one", significance test based on one family
                    "all", significance test based on all families

p_val_cutoff    :   p value cutoff, which is used to decide significant relationships 

tail_type       :   tells us the type of z_test we want to do
                    "positive", if p_val < p_val_cutoff and z_val > 0
                    "negative", if p_val < p_val_cutoff and z_val < 0
                    "both",     if p_val < p_val_cutoff

lag_L           :   list of lags, where each [lag[2 * i], lag[2 * i + 1]] is a time window needs to be tested
                    e.g. list of form: 1 5 5 10 represents windows of form: [1, 5] and [5, 10]
