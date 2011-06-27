#!/usr/bin/env python

import urllib
import string
import os
import sys
import LaunchOnCondor

FarmDirectory = "MERGE"

LaunchOnCondor.ListToFile(LaunchOnCondor.GetListOfFiles('"file:','/storage/data/cms/store/user/quertenmont/11_04_03_HSCP2011/FWLite_Data/RunA_*/*.root','",'), FarmDirectory + "InputFile.txt") 
LaunchOnCondor.SendCMSJobs(FarmDirectory, "Data_RunA", "Merge_cfg.py", FarmDirectory + "InputFile.txt", 1, [])
os.system("rm " +  FarmDirectory + "InputFile.txt")
