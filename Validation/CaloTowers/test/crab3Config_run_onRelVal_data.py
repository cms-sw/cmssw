from WMCore.Configuration import Configuration
config = Configuration()

config.section_("General")
config.General.requestName = 'JetHT_CMSSW_732_patch1'

config.section_("JobType")
config.JobType.pluginName = 'Analysis'
config.JobType.psetName = 'run_onRelVal_KH_cfg.py'
config.JobType.allowNonProductionCMSSW = False

config.section_("Data")
config.Data.inputDataset = '/JetHT/CMSSW_7_3_2_patch1-GR_R_73_V0_HcalExtValid_RelVal_jet2012D-v2/RECO'
config.Data.inputDBS = 'https://cmsweb.cern.ch/dbs/prod/global/DBSReader/'
config.Data.splitting = 'FileBased'
config.Data.unitsPerJob = 20
config.Data.publication = False
#config.Data.publishDBS = 'https://cmsweb.cern.ch/dbs/prod/phys03/DBSWriter/'
#config.Data.publishDataName = 'PHYS14_PU20bx25_PHYS14_25_V1-FLAT'
#Use your own username instead of the "lhx". Keep branch tag in the directory name, e.g., PHYS14_720_Dec23_2014.
config.Data.outLFN = '/store/user/hatake/DQMIO/'

config.Data.ignoreLocality = False

config.section_("Site")
config.Site.storageSite = 'T3_US_Baylor'
