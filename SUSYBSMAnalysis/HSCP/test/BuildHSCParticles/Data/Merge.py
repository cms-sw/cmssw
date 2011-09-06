#!/usr/bin/env python

import urllib
import string
import os
import sys
import LaunchOnCondor

FarmDirectory = "MERGE"

#LaunchOnCondor.ListToFile(LaunchOnCondor.GetListOfFiles('"','/store/user/farrell3/EDMFilesToBeMergedDTNewConstants/RunA_*_160404_163869/HSCP_*.root','",'), FarmDirectory + "InputFile.txt")
#LaunchOnCondor.SendCMSJobs(FarmDirectory, "Data_RunA_160404_163869", "Merge_cfg.py", FarmDirectory + "InputFile.txt", 1, [])

#LaunchOnCondor.ListToFile(LaunchOnCondor.GetListOfFiles('"','/store/user/farrell3/EDMFilesToBeMergedDTNewConstants/RunA_*_170826_171500/HSCP_*.root','",'), FarmDirectory + "InputFile.txt")
#LaunchOnCondor.SendCMSJobs(FarmDirectory, "Data_RunA_170826_171500", "Merge_cfg.py", FarmDirectory + "InputFile.txt", 1, [])

#LaunchOnCondor.ListToFile(LaunchOnCondor.GetListOfFiles('"','/store/user/farrell3/EDMFilesToBeMergedDTNewConstants/RunA_*_171501_172619/HSCP_*.root','",'), FarmDirectory + "InputFile.txt")
#LaunchOnCondor.SendCMSJobs(FarmDirectory, "Data_RunA_171501_172619", "Merge_cfg.py", FarmDirectory + "InputFile.txt", 1, [])

#LaunchOnCondor.ListToFile(LaunchOnCondor.GetListOfFiles('"','/store/user/farrell3/EDMFilesToBeMergedDTNewConstants/RunA_*_172620_172790/HSCP_*.root','",'), FarmDirectory + "InputFile.txt")
#LaunchOnCondor.SendCMSJobs(FarmDirectory, "Data_RunA_172620_172790", "Merge_cfg.py", FarmDirectory + "InputFile.txt", 1, [])

#LaunchOnCondor.ListToFile(LaunchOnCondor.GetListOfFiles('"','/store/user/farrell3/EDMFilesToBeMergedDTNewConstants/RunA_*_172791_172802/HSCP_*.root','",'), FarmDirectory + "InputFile.txt")
#LaunchOnCondor.SendCMSJobs(FarmDirectory, "Data_RunA_172791_172802", "Merge_cfg.py", FarmDirectory + "InputFile.txt", 1, [])

#LaunchOnCondor.ListToFile(LaunchOnCondor.GetListOfFiles('"','/store/user/farrell3/EDMFilesToBeMergedDTNewConstants/RunA_*_172803_172900/HSCP_*.root','",'), FarmDirectory + "InputFile.txt")
#LaunchOnCondor.SendCMSJobs(FarmDirectory, "Data_RunA_172803_172900", "Merge_cfg.py", FarmDirectory + "InputFile.txt", 1, [])

#LaunchOnCondor.ListToFile(LaunchOnCondor.GetListOfFiles('"','/store/user/farrell3/EDMFilesToBeMergedDTNewConstants/RunA_*_172901_173243/HSCP_*.root','",'), FarmDirectory + "InputFile.txt")
#LaunchOnCondor.SendCMSJobs(FarmDirectory, "Data_RunA_172901_173243", "Merge_cfg.py", FarmDirectory + "InputFile.txt", 1, [])

LaunchOnCondor.ListToFile(LaunchOnCondor.GetListOfFiles('"','/store/user/farrell3/HSCPEDMUpdateTo2p1fb01Sep2011/RunA_*_173244_173692/HSCP_*.root','",'), FarmDirectory + "InputFile.txt")
LaunchOnCondor.SendCMSJobs(FarmDirectory, "Data_RunA_173244_173692", "Merge_cfg.py", FarmDirectory + "InputFile.txt", 1, [])

os.system("rm " +  FarmDirectory + "InputFile.txt")
