(1) cmsrel CMSSW_7_6_0_pre4 (for instance)

(2) cd CMSSW_7_6_0_pre4/src

(3) cmsenv

(4) git cms-addpkg Validation/CaloTowers

(5) scram b

(6) cd Validation/CaloTowers/test/CaloScan

(7) ./make_configs.csh

(8) ./submit_batch.csh

NB: it uses batch submission (batch.csh) to lxbatch at CERN 
with input file 
/afs/cern.ch/cms/data/CMSSW/Validation/HcalHits/data/620/mc_pi50_eta05.root
Each of 25 job uses 2K out of total 50K input.

In 1-1.5 hour (in the submission directory, /scan in this case)
the results of 25 batch jobs will be arriving. 
Once all 25 jobs finished and 25 *.root files appeared locally, 

(9) ./Merge.sh 760pre3_postLS1

It will do following things:

cmsRun  merging_cfg.py
(to produce final DQMxxx.root file)

clean up the directory
rm -r pi50_*.py *.log LSFJOB_* pi50_*.root

Rename DQMxxx.root file to a convenient name

NB: there is naming convention  pi50scan<...>_ECALHCAL_CaloTowers.root
where <...> can be any meaningful string (to appear in the legend of histos). It generally indicates the release or upgrade status.

For example here it is 760pre3_postLS1 which need to be given as the argument for Merge.sh

move this pi50scan<...>_ECALHCAL_CaloTowers.root to Validation/CaloTowers/test/macros


(10) to compare two sets of histos, for instance if you have in 
Validation/CaloTowers/test/macros 
pi50scan760pre3_postLS1_ECALHCAL_CaloTowers.root    and
pi50scan760pre2_postLS1_ECALHCAL_CaloTowers.root

(here "760pre3_postLS1" and "760pre2_postLS1" are mentioned <...> strings) - 

./RunPions.csh 760pre3_postLS1 760pre2_postLS1


(11) the result appear as the local directory 
760pre3_postLS1_vs_760pre2_postLS1_SinglePi
  
which can be 
(i) viewed with web browser locally, e.g.
firefox 760pre3_postLS1_vs_760pre2_postLS1_SinglePi/index.html

(ii) uploaded to some web server and viewed from anywhere:

https://cms-cpt-software.web.cern.ch/cms-cpt-software/General/Validation/SVSuite/HCAL/calo_scan_single_pi/760pre3_postLS1_vs_760pre2_postLS1_SinglePi/

(in this case, the results from the two releases are just identical)
NB:  hitso labels correspond to aforementioned <...> strings, here 760pre3_postLS1/760pre2_postLS1

--------------------------------------------------
NB: recent changes in the template since 760pre6
to cope with a massive generator/smearing rearrangement in
https://github.com/cms-sw/cmssw/pull/10858

(A) to continue using default GEN source:
    /afs/cern.ch/cms/data/CMSSW/Validation/HcalHits/data/620/mc_pi50_eta05.root
to be used template.py_since760pre6 (renaming it back to template.py)

(B) for newly generated GEN:
    /afs/cern.ch/cms/data/CMSSW/Validation/HcalHits/data/76X/mc_pi50_eta05.root
to be used template.py_since760pre6_forGEN_760pre6 (renaming it back to template.py)
