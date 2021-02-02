from CRABAPI.RawCommand import crabCommand
from CRABClient.UserUtilities import config
from copy import deepcopy
import os
 
def submit(config):
    res = crabCommand('submit', config = config)
    #save crab config for the future
    with open(config.General.workArea + "/crab_" + config.General.requestName + "/crab_config.py", "w") as fi:
        fi.write(config.pythonise_())

samples = [
    ("/RelValQCD_Pt15To7000_Flat_14/CMSSW_11_3_0_pre1-113X_mcRun4_realistic_v1_2026D49noPU_rsb-v1/GEN-SIM-DIGI-RAW", "QCD_noPU_phase2"),
    ("/RelValQCD_Pt15To7000_Flat_14/CMSSW_11_3_0_pre1-PU_113X_mcRun4_realistic_v1_2026D49PU200-v1/GEN-SIM-DIGI-RAW", "QCD_PU_phase2"),
    ("/RelValZEE_14/CMSSW_11_3_0_pre1-PU_113X_mcRun4_realistic_v1_2026D49PU200-v1/GEN-SIM-DIGI-RAW", "ZEE_PU_phase2"),
    ("/RelValZMM_14/CMSSW_11_3_0_pre1-PU_113X_mcRun4_realistic_v1_2026D49PU200-v1/GEN-SIM-DIGI-RAW", "ZMM_PU_phase2"),
    ("/RelValTenTau_15_500_Eta3p1/CMSSW_11_3_0_pre1-PU_113X_mcRun4_realistic_v1_2026D49PU200-v1/GEN-SIM-DIGI-RAW", "TenTau_PU_phase2"),
    ("/RelValNuGun/CMSSW_11_3_0_pre1-PU_113X_mcRun4_realistic_v1_2026D49PU200-v1/GEN-SIM-DIGI-RAW", "NuGun_PU_phase2"),
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
        conf.JobType.maxJobRuntimeMin = 8*60
        conf.JobType.allowUndistributedCMSSW = True
        conf.JobType.outputFiles = ["step3_inMINIAODSIM.root"]
        conf.JobType.maxMemoryMB = 20000
        conf.JobType.numCores = 8
        
        conf.Data.inputDataset = dataset
        conf.Data.splitting = 'LumiBased'
        conf.Data.unitsPerJob = 5
        #conf.Data.totalUnits = 50
        conf.Data.publication = False
        conf.Data.outputDatasetTag = 'pfvalidation'
        #conf.Data.ignoreLocality = True
        
        # Where the output files will be transmitted to
        conf.Site.storageSite = 'T3_US_Baylor'
        #conf.Site.storageSite = 'T2_US_Caltech'
        #conf.Site.whitelist = ["T2_US_Caltech", "T2_CH_CERN"]
        
        submit(conf) 
