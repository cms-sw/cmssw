import subprocess
import argparse
import os
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Phase II Validation')
    parser.add_argument("inputdbsnames", type=str, nargs="+", help='input datasets')
    args = parser.parse_args()

    for dataset in args.inputdbsnames:
        outnamebase = "__".join(dataset.split("/")[:2])
        outname_dqm = outnamebase + "_DQM.root"
            
        out,err = subprocess.Popen(["cmsRun","Validation/HLTrigger/test/runPhaseIIValSource_cfg.py","inputFiles=dbs:"+dataset,f"outputFile={outname_dqm}"]).communicate()

        out,err = subprocess.Popen(["cmsRun","Validation/HLTrigger/test/runValClient_cfg.py","inputFiles={outname_dqm}"]).communicate()


        os.makedirs(outnamebase,exist_ok=True)
        os.rename("DQM_V0001_R000000001__HLT__Validation__All.root",outnamebase+"/"+"DQM_V0001_R000000001__HLT__Validation__All.root")

                                    
        