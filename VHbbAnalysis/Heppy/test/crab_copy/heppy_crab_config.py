from WMCore.Configuration import Configuration
config = Configuration()

config.section_("General")
config.General.requestName = 'VHBB_COPYMergeX30_V13_01'
config.General.workArea = 'crab_projects_V13_copy_2'
config.General.transferLogs=True

config.section_("JobType")
config.JobType.pluginName = 'Analysis'
config.JobType.psetName = 'heppy_crab_fake_pset.py'
config.JobType.scriptExe = 'copy_crab_script.sh'
config.JobType.inputFiles = ['copy_crab_script.py']
config.section_("Data")
config.Data.inputDataset = '/SingleMuon/arizzi-VHBB_HEPPY_V13_SingleMuon__Run2015C-PromptReco-v1-ec99a5ce649e4f29caa7e6ff25a1a44a/USER'
config.Data.inputDBS = 'phys03'
config.Data.splitting = 'FileBased'
config.Data.unitsPerJob = 30
config.Data.outLFNDirBase = '/store/user/jpata/VHBBHeppyV13/'
config.Data.publication = True
config.Data.publishDataName = 'VHBB_COPYMergeX30_V13'

config.section_("Site")
#config.Site.storageSite = "T2_CH_CERN"
#config.Site.storageSite = "T2_EE_Estonia"
config.Site.storageSite = "T3_CH_PSI"

#config.Site.whitelist = ["T2_IT_Pisa","T2_CH_CERN"]
config.Data.ignoreLocality = False #True
config.Site.whitelist = ["T2_IT_Pisa"]
