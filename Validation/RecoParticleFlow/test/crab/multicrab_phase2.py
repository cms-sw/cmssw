from CRABAPI.RawCommand import crabCommand
from CRABClient.UserUtilities import config
from copy import deepcopy
import os
import json

with open("../datasets_phase2.json") as f:
    dataset_configs = json.load(f)

def submit(config):
    res = crabCommand('submit', config = config)
    #save crab config for the future
    with open(config.General.workArea + "/crab_" + config.General.requestName + "/crab_config.py", "w") as fi:
        fi.write(config.pythonise_())

samples = [
    (d["path"], d["name"])
    for d in dataset_configs
    if d["name"] in ["QCD_noPU", "QCD_PU"] # submit only QCD_noPU QCD_PU
]

if __name__ == "__main__":
    for dataset, name in samples:

        if os.path.isfile("step3_phase2_dump.pyc"):
            os.remove("step3_phase2_dump.pyc")

        conf = config()

        conf.General.requestName = name
        conf.General.transferLogs = True
        conf.General.workArea = 'crab_projects_phase2'
        conf.JobType.pluginName = 'Analysis'
        conf.JobType.psetName = 'step3_phase2_dump.py'
        conf.JobType.maxJobRuntimeMin = 8*60
        conf.JobType.allowUndistributedCMSSW = True
        conf.JobType.outputFiles = ["step3_inMINIAODSIM.root"]
        conf.JobType.maxMemoryMB = 20000
        conf.JobType.numCores = 8

        conf.Data.inputDataset = dataset
        conf.Data.splitting = 'FileBased'
        conf.Data.unitsPerJob = 1
        #conf.Data.totalUnits = 50
        conf.Data.publication = False
        conf.Data.outputDatasetTag = 'pfvalidation'
        #conf.Data.ignoreLocality = True

        # Where the output files will be transmitted to
        conf.Site.storageSite = 'T3_US_Baylor'
        #conf.Site.storageSite = 'T2_US_Caltech'
        #conf.Site.whitelist = ["T2_US_Caltech", "T2_CH_CERN"]

        submit(conf)
