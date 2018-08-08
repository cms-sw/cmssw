How to use relval plotting machinary:

Example: CMSSW_10_2_0_pre6 vs CMSSW_10_2_0_pre5

Target relaease: CMSSW_10_2_0_pre6 , target GT: 102X_upgrade2018_realistic_v7
Reference relaease: CMSSW_10_2_0_pre5, reference GT: 102X_upgrade2018_realistic_v1


	cmsrel CMSSW_10_2_0
	cd CMSSW_10_2_0/src
	cmsenv

> download relval script
	
	git cms-addpkg Validation/CaloTowers
	### in case PR 24159 is not yet merged then use following command
	git cms-merge-topic spandeyehep:HCAL_DQM_hist_booking_change
	scram b
	cd Validation/CaloTowers/test/macros/
	make

> initialize proxies

	voms-proxy-init --voms cms
	### for csh shell
	setenv X509_USER_PROXY /tmp/x509up_u`id -u`
	### for bash shell
	export X509_USER_PROXY=/tmp/x509up_u`id -u`


> Download target and reference DQM files:

	### MC target
	./RelValHarvest.py -M CMSSW_10_2_0_pre6
	### MC reference
	./RelValHarvest.py -M CMSSW_10_2_0_pre5

> It will download and rename DQM files (the main files are following):

Target:
# HcalRecHitValidationRelVal_HighPtQCD_1020pre6_102X_upgrade2018_realistic_v7-v1.root
# HcalRecHitValidationRelVal_MinBias_1020pre6_102X_upgrade2018_realistic_v7-v1.root
# HcalRecHitValidationRelVal_QCD_1020pre6_102X_upgrade2018_realistic_v7-v1.root
# HcalRecHitValidationRelVal_TTbar_1020pre6_102X_upgrade2018_realistic_v7-v1.root
# HcalRecHitValidationRelVal_TTbar_1020pre6_PU25ns_102X_upgrade2018_realistic_v7_rsb-v1.root
# HcalRecHitValidationRelVal_TTbar_1020pre6_PUpmx25ns_102X_upgrade2018_realistic_v7-v1.root

Reference:
# HcalRecHitValidationRelVal_HighPtQCD_1020pre5_102X_upgrade2018_realistic_v1-v1.root
# HcalRecHitValidationRelVal_MinBias_1020pre5_102X_upgrade2018_realistic_v1-v1.root
# HcalRecHitValidationRelVal_QCD_1020pre5_102X_upgrade2018_realistic_v1-v1.root
# HcalRecHitValidationRelVal_TTbar_1020pre5_102X_upgrade2018_realistic_v1-v1.root
# HcalRecHitValidationRelVal_TTbar_1020pre5_PU25ns_102X_upgrade2018_realistic_v1-v1.root
# HcalRecHitValidationRelVal_TTbar_1020pre5_PUpmx25ns_102X_upgrade2018_realistic_v1-v1.root



> Generate plots: We'll need to generate plots for standard MC samples, PU25ns samples & PUpmx25ns samples:

	### MC samples
	./RunRVMacros2018.csh 1020pre6_102X_upgrade2018_realistic_v7-v1 1020pre5_102X_upgrade2018_realistic_v1-v1
	### PU25ns samples
	./RunRVMacros_Pileup2018.csh 1020pre6_PU25ns_102X_upgrade2018_realistic_v7-v1 1020pre5_PU25ns_102X_upgrade2018_realistic_v1-v1

It will generate the plots for all the samples.




>>> Similarly for DATA:

> Download target and reference DQM files (2017B):

	### Data target
	./RelValHarvest.py -D CMSSW_10_2_0_pre6
	### Data reference
	./RelValHarvest.py -D CMSSW_10_2_0_pre5

> It will download and rename DQM files (the main files are following):

Target:
# HcalRecHitValidationRelVal_ZeroBias_1020pre6_102X_dataRun2_PromptLike_v3.root
# HcalRecHitValidationRelVal_JetHT_1020pre6_102X_dataRun2_PromptLike_v3.root

Reference:
# HcalRecHitValidationRelVal_ZeroBias_1020pre5_102X_dataRun2_PromptLike_v3.root
# HcalRecHitValidationRelVal_JetHT_1020pre5_102X_dataRun2_PromptLike_v3.root

> Generate plots:

	./RunRVMacros_DATA.csh 1020pre6_102X_dataRun2_PromptLike_v3 1020pre5_102X_dataRun2_PromptLike_v3


> upload the plots

	rsync -av 1020* userId@lxplus.cern.ch:/afs/cern.ch/cms/cpt/Software/html/General/Validation/SVSuite/HCAL/

NOTE: To download 2018A or 2018B samples, use RelValHarvest_2018A.py or RelValHarvest_2018B.py script.
