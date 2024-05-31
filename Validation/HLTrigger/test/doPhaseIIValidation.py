import subprocess
import argparse
import os

def lookup_datasets(sample_tag):

    primary_datasets = ["RelValZpToMM_m6000_14TeV","RelValZpToEE_m6000_14TeV","RelValZpTT_1500_14","RelValTTbarToDilepton_14TeV","RelValQCD_Pt15To7000_Flat_14","RelValTTbar_14TeV"]
    return [f"/{p}/{sample_tag}/GEN-SIM-RECO" for p in primary_datasets]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Phase II Validation')
    parser.add_argument("--inputdbsnames", type=str, nargs="+", default=None, help='input datasets')
    parser.add_argument("--sampletag", type=str, default=None, help='sample tag')
    
    args = parser.parse_args()


    if args.inputdbsnames is None:
        datasets = lookup_datasets(args.sampletag)
    else:
        datasets = args.inputdbsnames
    

    for dataset in datasets:
        outnamebase = "__".join(dataset.split("/")[1:3])
        outname_dqm = outnamebase + "_DQM.root"
            
        out,err = subprocess.Popen(["cmsRun","Validation/HLTrigger/test/runPhaseIIValSource_cfg.py","inputFiles=dbs:"+dataset,f"outputFile={outname_dqm}",f"sampleLabel={args.sampletag}"]).communicate()

        out,err = subprocess.Popen(["cmsRun","Validation/HLTrigger/test/runValClient_cfg.py",f"inputFiles=file:{outname_dqm}"]).communicate()


        os.makedirs(outnamebase,exist_ok=True)
        os.rename("DQM_V0001_R000000001__HLT__Validation__All.root",outnamebase+"/"+"DQM_V0001_R000000001__HLT__Validation__All.root")

                                    
        
