from CRABClient.UserUtilities import config
config = config()

config.General.requestName = 'test1'
config.General.transferLogs = True

config.JobType.pluginName = 'Analysis'
config.JobType.psetName = 'step3.py'
config.JobType.maxJobRuntimeMin = 2*60
#config.JobType.allowUndistributedCMSSW = True
config.JobType.outputFiles = ["step3_inMINIAODSIM.root"]

config.Data.inputDataset = '/RelValQCD_FlatPt_15_3000HS_13/CMSSW_10_4_0_pre4-103X_mc2017_realistic_v2-v1/GEN-SIM-DIGI-RAW'
config.Data.splitting = 'LumiBased'
config.Data.unitsPerJob = 500/100
config.Data.publication = False
config.Data.outputDatasetTag = 'CRAB3_Analysis_test1'
config.Data.ignoreLocality = True

# Where the output files will be transmitted to
config.Site.storageSite = 'T2_US_Caltech'
config.Site.whitelist = ["T2_US_*", "T2_CH_*"]
