#!/usr/bin/env python

import urllib
import string
import os
import sys
import LaunchOnCondor

Jobs = ["GMStau_8TeV_M557", "PPStau_8TeV_M557"]

FarmDirectory = "MERGE"
for JobName in Jobs:
	LaunchOnCondor.ListToFile(LaunchOnCondor.GetListOfFiles('"file:','/afs/cern.ch/user/q/querten/workspace/public/GMSB_XSec/EDMproduction/CMSSW_5_3_2_patch4/src/SUSYBSMAnalysis/HSCP/test/BuildHSCParticles/Signals/'+ JobName + '/res/HSCP_*.root','",'), FarmDirectory + "InputFile.txt")
#	LaunchOnCondor.ListToFile(LaunchOnCondor.GetListOfFiles('"dcache:','/pnfs/cms/WAX/11/store/user/farrell3/store/user/jchen/11_10_28_HSCP2011/FWLite_Signal/'+ JobName + '/HSCP_*.root','",'), FarmDirectory + "InputFile.txt")
#	LaunchOnCondor.ListToFile(LaunchOnCondor.GetListOfFiles('"file:','/uscmst1b_scratch/lpc1/lpcphys/jchen/HSCPEDM_08_02_11/'+ JobName + '.root','",'), FarmDirectory + "InputFile.txt") 
	LaunchOnCondor.SendCMSJobs(FarmDirectory, JobName, "Merge_cfg.py", FarmDirectory + "InputFile.txt", 1, [])

os.system("rm " +  FarmDirectory + "InputFile.txt")
