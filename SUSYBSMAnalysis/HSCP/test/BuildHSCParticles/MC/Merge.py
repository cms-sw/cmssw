#!/usr/bin/env python

import urllib
import string
import os
import sys
import LaunchOnCondor





Jobs = ["DYToMuMu","DYToTauTau","QCD_Pt-1000to1400","QCD_Pt-120to170","QCD_Pt-1400to1800","QCD_Pt-15to30","QCD_Pt-170to300","QCD_Pt-1800","QCD_Pt-300to470","QCD_Pt-30to50","QCD_Pt-470to600","QCD_Pt-50to80","QCD_Pt-600to800","QCD_Pt-800to1000","QCD_Pt-80to120","TTJets","WJetsToLNu","WToMuNu","WToTauNu","ZJetToMuMu_Pt-0to15","ZJetToMuMu_Pt-120to170","ZJetToMuMu_Pt-15to20","ZJetToMuMu_Pt-170to230","ZJetToMuMu_Pt-20to30","ZJetToMuMu_Pt-230to300","ZJetToMuMu_Pt-300","ZJetToMuMu_Pt-30to50","ZJetToMuMu_Pt-50to80","ZJetToMuMu_Pt-80to120", "WW", "WZ", "ZZ"]

FarmDirectory = "MERGE"
for JobName in Jobs:
	LaunchOnCondor.ListToFile(LaunchOnCondor.GetListOfFiles('"file:','/storage/data/cms/store/user/quertenmont/11_07_31_HSCP2011/FWLite_MC/'+ JobName + '/HSCP_*.root','",'), FarmDirectory + "InputFile.txt") 
	LaunchOnCondor.SendCMSJobs(FarmDirectory, "MC_" + JobName, "Merge_cfg.py", FarmDirectory + "InputFile.txt", 1, [])

os.system("rm " +  FarmDirectory + "InputFile.txt")
