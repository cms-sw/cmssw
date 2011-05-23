#!/usr/bin/env python

import urllib
import string
import os
import sys
import LaunchOnCondor

FarmDirectory = "MERGENEW"

LaunchOnCondor.ListToFile(LaunchOnCondor.GetListOfFiles('"file:','/storage/data/cms/store/user/quertenmont/11_05_16_HSCP2011/FWLite_Data/RunA_*V1/*.root','",'), FarmDirectory + "InputFile.txt") 
LaunchOnCondor.SendCMSJobs(FarmDirectory, "Data_RunA_V1", "Merge_cfg.py", FarmDirectory + "InputFile.txt", 1, [])

LaunchOnCondor.ListToFile(LaunchOnCondor.GetListOfFiles('"file:','/storage/data/cms/store/user/quertenmont/11_05_16_HSCP2011/FWLite_Data/RunA_*V2a/*.root','",'), FarmDirectory + "InputFile.txt")
LaunchOnCondor.SendCMSJobs(FarmDirectory, "Data_RunA_V2a", "Merge_cfg.py", FarmDirectory + "InputFile.txt", 1, [])

LaunchOnCondor.ListToFile(LaunchOnCondor.GetListOfFiles('"file:','/storage/data/cms/store/user/quertenmont/11_05_16_HSCP2011/FWLite_Data/RunA_*V2b/*.root','",'), FarmDirectory + "InputFile.txt")
LaunchOnCondor.SendCMSJobs(FarmDirectory, "Data_RunA_V2b", "Merge_cfg.py", FarmDirectory + "InputFile.txt", 1, [])

LaunchOnCondor.ListToFile(LaunchOnCondor.GetListOfFiles('"file:','/storage/data/cms/store/user/quertenmont/11_05_16_HSCP2011/FWLite_Data/RunA_*V2c/*.root','",'), FarmDirectory + "InputFile.txt")
LaunchOnCondor.SendCMSJobs(FarmDirectory, "Data_RunA_V2c", "Merge_cfg.py", FarmDirectory + "InputFile.txt", 1, [])

LaunchOnCondor.ListToFile(LaunchOnCondor.GetListOfFiles('"file:','/storage/data/cms/store/user/quertenmont/11_05_16_HSCP2011/FWLite_Data/RunA_*V2d/*.root','",'), FarmDirectory + "InputFile.txt")
LaunchOnCondor.SendCMSJobs(FarmDirectory, "Data_RunA_V2d", "Merge_cfg.py", FarmDirectory + "InputFile.txt", 1, [])

LaunchOnCondor.ListToFile(LaunchOnCondor.GetListOfFiles('"file:','/storage/data/cms/store/user/quertenmont/11_05_16_HSCP2011/FWLite_Data/RunA_*V2e/*.root','",'), FarmDirectory + "InputFile.txt")
LaunchOnCondor.SendCMSJobs(FarmDirectory, "Data_RunA_V2e", "Merge_cfg.py", FarmDirectory + "InputFile.txt", 1, [])

os.system("rm " +  FarmDirectory + "InputFile.txt")
