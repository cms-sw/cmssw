from CRABAPI.RawCommand import crabCommand
from CRABClient.UserUtilities import getUsernameFromSiteDB
from CRABClient.UserUtilities import config
from copy import deepcopy
import os
 
def submit(config):
    res = crabCommand('submit', config = config)
    #save crab config for the future
    with open(config.General.workArea + "/crab_" + config.General.requestName + "/crab_config.py", "w") as fi:
        fi.write(config.pythonise_())

samples = [
    ("/RelValQCD_FlatPt_15_3000HS_13/CMSSW_10_4_0_pre4-103X_mc2017_realistic_v2-v1/GEN-SIM-DIGI-RAW", "QCD_FlatPt_noPU"),
    ("/RelValQCD_FlatPt_15_3000HS_13/CMSSW_10_4_0_pre4-PU25ns_103X_mc2017_realistic_v2-v1/GEN-SIM-DIGI-RAW", "QCD_FlatPt_PU25ns"),
    ("/RelValZMM_13/CMSSW_10_4_0_pre4-103X_mc2017_realistic_v2-v1/GEN-SIM-DIGI-RAW", "ZMM"),
    ("/RelValMinBias_13/CMSSW_10_4_0_pre4-103X_mc2017_realistic_v2-v1/GEN-SIM-DIGI-RAW", "MinBias"),
]

if __name__ == "__main__":
    for dataset, name in samples:

        if os.path.isfile("step3_dump.pyc"):
            os.remove("step3_dump.pyc")
 
        conf = config()
        
        conf.General.requestName = name
        conf.General.transferLogs = True
        conf.General.workArea = 'crab_projects'
        conf.JobType.pluginName = 'Analysis'
        conf.JobType.psetName = 'step3_dump.py'
        conf.JobType.maxJobRuntimeMin = 4*60
        conf.JobType.allowUndistributedCMSSW = True
        conf.JobType.outputFiles = ["step3_inMINIAODSIM.root"]
        conf.JobType.maxMemoryMB = 4000
        conf.JobType.numCores = 2
        
        conf.Data.inputDataset = dataset
        conf.Data.splitting = 'LumiBased'
        conf.Data.unitsPerJob = 10
        #conf.Data.totalUnits = 50
        conf.Data.publication = False
        conf.Data.outputDatasetTag = 'pfvalidation'
        #conf.Data.ignoreLocality = True
        
        # Where the output files will be transmitted to
        conf.Site.storageSite = 'T2_US_Caltech'
        #conf.Site.whitelist = ["T2_US_Caltech", "T2_CH_CERN"]
        
        submit(conf) 
