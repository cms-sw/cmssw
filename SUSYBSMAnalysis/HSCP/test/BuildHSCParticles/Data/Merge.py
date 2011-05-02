#!/usr/bin/env python

import urllib
import string
import os
import sys
import LaunchOnCondor

FarmDirectory = "MERGE"

LaunchOnCondor.ListToFile(LaunchOnCondor.GetListOfFiles('"file:','/storage/data/cms/store/user/quertenmont/11_04_30v3_HSCP2011/FWLite_Data/RunA_*V1/*.root','",'), FarmDirectory + "InputFile.txt") 
LaunchOnCondor.SendCMSJobs(FarmDirectory, "Data_RunA_V1", "Merge_cfg.py", FarmDirectory + "InputFile.txt", 1, [])

LaunchOnCondor.ListToFile(LaunchOnCondor.GetListOfFiles('"file:','/storage/data/cms/store/user/quertenmont/11_04_30v3_HSCP2011/FWLite_Data/RunA_*V2/*.root','",'), FarmDirectory + "InputFile.txt")
LaunchOnCondor.SendCMSJobs(FarmDirectory, "Data_RunA_V2", "Merge_cfg.py", FarmDirectory + "InputFile.txt", 1, [])


os.system("rm " +  FarmDirectory + "InputFile.txt")
