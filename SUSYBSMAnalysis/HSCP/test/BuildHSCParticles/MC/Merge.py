#!/usr/bin/env python

import urllib
import string
import os
import sys
import LaunchOnCondor





#Jobs = ["MC_7TeV_ZZ", "MC_7TeV_ZJetToMuMu_Pt-80to120", "MC_7TeV_ZJetToMuMu_Pt-50to80", "MC_7TeV_ZJetToMuMu_Pt-30to50", "MC_7TeV_ZJetToMuMu_Pt-300", "MC_7TeV_ZJetToMuMu_Pt-230to300", "MC_7TeV_ZJetToMuMu_Pt-20to30", "MC_7TeV_ZJetToMuMu_Pt-170to230", "MC_7TeV_ZJetToMuMu_Pt-15to20", "MC_7TeV_ZJetToMuMu_Pt-120to170", "MC_7TeV_ZJetToMuMu_Pt-0to15", "MC_7TeV_WZ", "MC_7TeV_WW", "MC_7TeV_WJetsToLNu", "MC_7TeV_TTJets", "MC_7TeV_QCD_Pt-80to120", "MC_7TeV_QCD_Pt-800to1000", "MC_7TeV_QCD_Pt-600to800", "MC_7TeV_QCD_Pt-50to80", "MC_7TeV_QCD_Pt-470to600", "MC_7TeV_QCD_Pt-30to50", "MC_7TeV_QCD_Pt-300to470", "MC_7TeV_QCD_Pt-1800", "MC_7TeV_QCD_Pt-170to300", "MC_7TeV_QCD_Pt-1400to1800", "MC_7TeV_QCD_Pt-120to170", "MC_7TeV_QCD_Pt-1000to1400", "MC_7TeV_DYToTauTau", "MC_7TeV_DYToMuMu"]

Jobs = ["MC_8TeV_DYToMuMu"]

FarmDirectory = "MERGE"
for JobName in Jobs:
        LaunchOnCondor.ListToFile(LaunchOnCondor.GetListOfFiles('"file:','/storage/data/cms/store/user/quertenmont/12_08_16_HSCP_EDM2011/FWLite_MC/' + JobName + '/HSCP_*.root','",'), FarmDirectory + "InputFile.txt")
	#LaunchOnCondor.ListToFile(LaunchOnCondor.GetListOfFiles('"dcache:','/pnfs/cms/WAX/11/store/user/farrell3/HSCPEDMUpdateData2012_12Sep2012/'+ JobName +'/HSCP_*.root','",'), FarmDirectory + "InputFile.txt")
	LaunchOnCondor.SendCMSJobs(FarmDirectory, JobName, "Merge_cfg.py", FarmDirectory + "InputFile.txt", 1, ['XXX_SAVEPATH_XXX','file:/storage/data/cms/users/quertenmont/HSCP/CMSSW_4_2_8/12_08_16/'])


os.system("rm " +  FarmDirectory + "InputFile.txt")
