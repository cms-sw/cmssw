from CRABAPI.RawCommand import crabCommand
from CRABClient.UserUtilities import config
from copy import deepcopy
import os
import json
import argparse

AVAILABLE_DATASETS = [
    "QCD_noPU",
    "QCD_PU",
    "ZEE_PU",
    "ZMM_PU",
    "TenTau_PU",
    "NuGun_PU"
]

#
# add arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--pset",
    default="step3_dump.py",
    help="CMSSW config to run (e.g. step3_dump.py or step3_pf_dump.py)"
)
parser.add_argument(
    "--workArea",
    default="crab_projects",
    help="CRAB work area"
)
parser.add_argument(
    "--gpu",
    action="store_true",
    help="Request GPU resources"
)
parser.add_argument(
    "--datasets",
    nargs="+",
    default=["QCD_noPU"],
    help="Datasets to submit (space-separated). "
         "Available: " + " ".join(AVAILABLE_DATASETS)
)
args = parser.parse_args()

# validate input datasets
invalid = [d for d in args.datasets if d not in AVAILABLE_DATASETS]
if invalid:
    parser.error(
        "Invalid dataset(s): {}. Available datasets: {}".format(
            ", ".join(invalid),
            " ".join(AVAILABLE_DATASETS)
        )
    )

# load datasets
with open("../datasets.json") as f:
    dataset_configs = json.load(f)

def submit(config):
    res = crabCommand('submit', config = config)
    #save crab config for the future
    with open(config.General.workArea + "/crab_" + config.General.requestName + "/crab_config.py", "w") as fi:
        fi.write(config.pythonise_())

samples = [
    (d["path"], d["name"])
    for d in dataset_configs
    if d["name"] in args.datasets
]

if __name__ == "__main__":
    for dataset, name in samples:

        if os.path.isfile(args.pset + "c"):   # remove .pyc if present
            os.remove(args.pset + "c")
            
        conf = config()

        conf.General.requestName = name
        conf.General.transferLogs = True
        conf.General.workArea = args.workArea        
        conf.JobType.pluginName = 'Analysis'
        conf.JobType.psetName = args.pset
        conf.JobType.maxJobRuntimeMin = 8*60
        conf.JobType.allowUndistributedCMSSW = True
        conf.JobType.outputFiles = ["step3_inMINIAODSIM.root"]
        conf.JobType.maxMemoryMB = 16000
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
        #conf.Site.whitelist = ["T2_US_Caltech", "T2_CH_CERN", "T3_US_Baylor"]

        if args.gpu:
            conf.Site.requireAccelerator = True
            #conf.Site.acceleratorParams = {
            #    "GPUMemoryMB": "4000",
            #    "GPUMinimumCapability": "7.0",
            #    "GPUMaximumCapability": "8.0",
            #    "GPURuntime": "12.1",
            #}
            # sites with GPUs are limited. use ignoreLocality.            
            conf.Data.ignoreLocality = True            
            conf.Site.whitelist = ["T2_US_Caltech", "T2_CH_CERN", "T2_US_Purdue", "T2_US_Wisconsin", "T2_UK_SGrid_RALPP", "T2_US_Caltech", "T1_ES_PIC", "T2_US_Florida", "T1_DE_KIT"]

        submit(conf)
