#!/usr/bin/env python

import urllib
import string
import os
import sys
import LaunchOnCondor

FarmDirectory = "MERGE"

LaunchOnCondor.ListToFile(LaunchOnCondor.GetListOfFiles('"file:','/storage/data/cms/store/user/quertenmont/11_07_02_HSCP2011/FWLite_Data/RunA_*_160404_163869/*.root','",'), FarmDirectory + "InputFile.txt")
LaunchOnCondor.SendCMSJobs(FarmDirectory, "Data_RunA_160404_163869", "Merge_cfg.py", FarmDirectory + "InputFile.txt", 1, [])

LaunchOnCondor.ListToFile(LaunchOnCondor.GetListOfFiles('"file:','/storage/data/cms/store/user/quertenmont/11_07_02_HSCP2011/FWLite_Data/RunA_*_165001_166033/*.root','",'), FarmDirectory + "InputFile.txt")
LaunchOnCondor.SendCMSJobs(FarmDirectory, "Data_RunA_165001_166033", "Merge_cfg.py", FarmDirectory + "InputFile.txt", 1, [])

LaunchOnCondor.ListToFile(LaunchOnCondor.GetListOfFiles('"file:','/storage/data/cms/store/user/quertenmont/11_07_02_HSCP2011/FWLite_Data/RunA_*_166034_166500/*.root','",'), FarmDirectory + "InputFile.txt")
LaunchOnCondor.SendCMSJobs(FarmDirectory, "Data_RunA_166034_166500", "Merge_cfg.py", FarmDirectory + "InputFile.txt", 1, [])

LaunchOnCondor.ListToFile(LaunchOnCondor.GetListOfFiles('"file:','/storage/data/cms/store/user/quertenmont/11_07_02_HSCP2011/FWLite_Data/RunA_*_166501_166893/*.root','",'), FarmDirectory + "InputFile.txt")
LaunchOnCondor.SendCMSJobs(FarmDirectory, "Data_RunA_166501_166893", "Merge_cfg.py", FarmDirectory + "InputFile.txt", 1, [])

LaunchOnCondor.ListToFile(LaunchOnCondor.GetListOfFiles('"file:','/storage/data/cms/store/user/quertenmont/11_07_02_HSCP2011/FWLite_Data/RunA_*_166894_167151/*.root','",'), FarmDirectory + "InputFile.txt")
LaunchOnCondor.SendCMSJobs(FarmDirectory, "Data_RunA_166894_167151", "Merge_cfg.py", FarmDirectory + "InputFile.txt", 1, [])

LaunchOnCondor.ListToFile(LaunchOnCondor.GetListOfFiles('"file:','/storage/data/cms/store/user/quertenmont/11_07_02_HSCP2011/FWLite_Data/RunA_*_167153_167784/*.root','",'), FarmDirectory + "InputFile.txt")
LaunchOnCondor.SendCMSJobs(FarmDirectory, "Data_RunA_167153_167784", "Merge_cfg.py", FarmDirectory + "InputFile.txt", 1, [])

os.system("rm " +  FarmDirectory + "InputFile.txt")
