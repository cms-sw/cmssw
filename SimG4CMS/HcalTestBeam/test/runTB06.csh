#!/bin/csh

python SimG4CMS/HcalTestBeam/test/python/run_tb06_all_cfg.py FTFP_BERT_EMM pi- $1 RR
python SimG4CMS/HcalTestBeam/test/python/run_tb06_all_cfg.py FTFP_BERT_EMM kaon+ $1 RR
python SimG4CMS/HcalTestBeam/test/python/run_tb06_all_cfg.py FTFP_BERT_EMM pbar $1 RR
python SimG4CMS/HcalTestBeam/test/python/run_tb06_all_cfg.py FTFP_BERT_EMM p $1 RR
python SimG4CMS/HcalTestBeam/test/python/run_tb06_all_cfg.py FTFP_BERT_EMM pi+ $1 RR
python SimG4CMS/HcalTestBeam/test/python/run_tb06_all_cfg.py FTFP_BERT_EMM kaon- $1 RR


#
