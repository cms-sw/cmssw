#!/usr/bin/env python

import urllib
import string
import os
import sys
import LaunchOnCondor

FarmDirectory = "MERGE"

#LaunchOnCondor.ListToFile(LaunchOnCondor.GetListOfFiles('"file:','/storage/data/cms/store/user/quertenmont/11_06_03_HSCP2011/FWLite_Data/RunA_*_160000_163250/*.root','",'), FarmDirectory + "InputFile.txt") 
#LaunchOnCondor.SendCMSJobs(FarmDirectory, "Data_RunA_160000_163250", "Merge_cfg.py", FarmDirectory + "InputFile.txt", 1, [])

#LaunchOnCondor.ListToFile(LaunchOnCondor.GetListOfFiles('"file:','/storage/data/cms/store/user/quertenmont/11_06_03_HSCP2011/FWLite_Data/RunA_*_163251_163500/*.root','",'), FarmDirectory + "InputFile.txt")
#LaunchOnCondor.SendCMSJobs(FarmDirectory, "Data_RunA_163251_163500", "Merge_cfg.py", FarmDirectory + "InputFile.txt", 1, [])

#LaunchOnCondor.ListToFile(LaunchOnCondor.GetListOfFiles('"file:','/storage/data/cms/store/user/quertenmont/11_06_03_HSCP2011/FWLite_Data/RunA_*_163501_164000/*.root','",'), FarmDirectory + "InputFile.txt")
#LaunchOnCondor.SendCMSJobs(FarmDirectory, "Data_RunA_163501_164000", "Merge_cfg.py", FarmDirectory + "InputFile.txt", 1, [])


LaunchOnCondor.ListToFile(LaunchOnCondor.GetListOfFiles('"file:','/storage/data/cms/store/user/quertenmont/11_06_22_HSCP2011/FWLite_Data/RunA_*_165001_166033/*.root','",'), FarmDirectory + "InputFile.txt")
LaunchOnCondor.SendCMSJobs(FarmDirectory, "Data_RunA_165001_166033", "Merge_cfg.py", FarmDirectory + "InputFile.txt", 1, [])

LaunchOnCondor.ListToFile(LaunchOnCondor.GetListOfFiles('"file:','/storage/data/cms/store/user/quertenmont/11_06_22_HSCP2011/FWLite_Data/RunA_*_166034_166500/*.root','",'), FarmDirectory + "InputFile.txt")
LaunchOnCondor.SendCMSJobs(FarmDirectory, "Data_RunA_166034_166500", "Merge_cfg.py", FarmDirectory + "InputFile.txt", 1, [])

LaunchOnCondor.ListToFile(LaunchOnCondor.GetListOfFiles('"file:','/storage/data/cms/store/user/quertenmont/11_06_22_HSCP2011/FWLite_Data/RunA_*_166501_166893/*.root','",'), FarmDirectory + "InputFile.txt")
LaunchOnCondor.SendCMSJobs(FarmDirectory, "Data_RunA_166501_166893", "Merge_cfg.py", FarmDirectory + "InputFile.txt", 1, [])

LaunchOnCondor.ListToFile(LaunchOnCondor.GetListOfFiles('"file:','/storage/data/cms/store/user/quertenmont/11_06_22_HSCP2011/FWLite_Data/RunA_*_166894_167151/*.root','",'), FarmDirectory + "InputFile.txt")
LaunchOnCondor.SendCMSJobs(FarmDirectory, "Data_RunA_166894_167151", "Merge_cfg.py", FarmDirectory + "InputFile.txt", 1, [])



os.system("rm " +  FarmDirectory + "InputFile.txt")
