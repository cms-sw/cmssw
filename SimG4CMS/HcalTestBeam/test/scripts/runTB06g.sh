#!/bin/bash

python SimG4CMS/HcalTestBeam/test/python/run_tb06_all_cfg.py FTFP_BERT_EMM gamma 100 RR >& gamma.out &
python SimG4CMS/HcalTestBeam/test/python/run_tb06_all_cfg.py FTFP_BERT_EMM e+ 100 RR >& e+.out &

#
