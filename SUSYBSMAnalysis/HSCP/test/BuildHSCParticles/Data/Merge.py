#!/usr/bin/env python

import urllib
import string
import os
import sys
import LaunchOnCondor

FarmDirectory = "MERGE"

#LaunchOnCondor.ListToFile(LaunchOnCondor.GetListOfFiles('"','/store/user/farrell3/HSCPEDMUpdateTo4p3fb28Oct2011/RunA_*_160404_163869/HSCP_*.root','",'), FarmDirectory + "InputFile.txt")
#LaunchOnCondor.SendCMSJobs(FarmDirectory, "Data_RunA_160404_163869", "Merge_cfg.py", FarmDirectory + "InputFile.txt", 1, [])

#LaunchOnCondor.ListToFile(LaunchOnCondor.GetListOfFiles('"','/store/user/farrell3/HSCPEDMUpdateTo4p3fb28Oct2011/RunA_*_165001_166033/HSCP_*.root','",'), FarmDirectory + "InputFile.txt")
#LaunchOnCondor.SendCMSJobs(FarmDirectory, "Data_RunA_165001_166033", "Merge_cfg.py", FarmDirectory + "InputFile.txt", 1, [])

#LaunchOnCondor.ListToFile(LaunchOnCondor.GetListOfFiles('"','/store/user/farrell3/HSCPEDMUpdateTo4p3fb28Oct2011/RunA_*_166034_166500/HSCP_*.root','",'), FarmDirectory + "InputFile.txt")
#LaunchOnCondor.SendCMSJobs(FarmDirectory, "Data_RunA_166034_166500", "Merge_cfg.py", FarmDirectory + "InputFile.txt", 1, [])

#LaunchOnCondor.ListToFile(LaunchOnCondor.GetListOfFiles('"','/store/user/farrell3/HSCPEDMUpdateTo4p3fb28Oct2011/RunA_*_166501_166893/HSCP_*.root','",'), FarmDirectory + "InputFile.txt")
#LaunchOnCondor.SendCMSJobs(FarmDirectory, "Data_RunA_166501_166893", "Merge_cfg.py", FarmDirectory + "InputFile.txt", 1, [])

#LaunchOnCondor.ListToFile(LaunchOnCondor.GetListOfFiles('"','/store/user/farrell3/HSCPEDMUpdateTo4p3fb28Oct2011/RunA_*_166894_167151/HSCP_*.root','",'), FarmDirectory + "InputFile.txt")
#LaunchOnCondor.SendCMSJobs(FarmDirectory, "Data_RunA_166894_167151", "Merge_cfg.py", FarmDirectory + "InputFile.txt", 1, [])

#LaunchOnCondor.ListToFile(LaunchOnCondor.GetListOfFiles('"','/store/user/farrell3/HSCPEDMUpdateTo4p3fb28Oct2011/RunA_*_167153_167913/HSCP_*.root','",'), FarmDirectory + "InputFile.txt")
#LaunchOnCondor.SendCMSJobs(FarmDirectory, "Data_RunA_167153_167913", "Merge_cfg.py", FarmDirectory + "InputFile.txt", 1, [])

#LaunchOnCondor.ListToFile(LaunchOnCondor.GetListOfFiles('"','/store/user/farrell3/HSCPEDMUpdateTo4p3fb28Oct2011/RunA_*_170826_171500/HSCP_*.root','",'), FarmDirectory + "InputFile.txt")
#LaunchOnCondor.SendCMSJobs(FarmDirectory, "Data_RunA_170826_171500", "Merge_cfg.py", FarmDirectory + "InputFile.txt", 1, [])

#LaunchOnCondor.ListToFile(LaunchOnCondor.GetListOfFiles('"','/store/user/farrell3/HSCPEDMUpdateTo4p3fb28Oct2011/RunA_*_171501_172619/HSCP_*.root','",'), FarmDirectory + "InputFile.txt")
#LaunchOnCondor.SendCMSJobs(FarmDirectory, "Data_RunA_171501_172619", "Merge_cfg.py", FarmDirectory + "InputFile.txt", 1, [])

#LaunchOnCondor.ListToFile(LaunchOnCondor.GetListOfFiles('"','/store/user/farrell3/HSCPEDMUpdateTo4p3fb28Oct2011/RunA_*_172620_172790/HSCP_*.root','",'), FarmDirectory + "InputFile.txt")
#LaunchOnCondor.SendCMSJobs(FarmDirectory, "Data_RunA_172620_172790", "Merge_cfg.py", FarmDirectory + "InputFile.txt", 1, [])

#LaunchOnCondor.ListToFile(LaunchOnCondor.GetListOfFiles('"','/store/user/farrell3/HSCPEDMUpdateTo4p3fb28Oct2011/RunA_*_172791_172802/HSCP_*.root','",'), FarmDirectory + "InputFile.txt")
#LaunchOnCondor.SendCMSJobs(FarmDirectory, "Data_RunA_172791_172802", "Merge_cfg.py", FarmDirectory + "InputFile.txt", 1, [])

#LaunchOnCondor.ListToFile(LaunchOnCondor.GetListOfFiles('"','/store/user/farrell3/HSCPEDMUpdateTo4p3fb28Oct2011/RunA_*_172803_172900/HSCP_*.root','",'), FarmDirectory + "InputFile.txt")
#LaunchOnCondor.SendCMSJobs(FarmDirectory, "Data_RunA_172803_172900", "Merge_cfg.py", FarmDirectory + "InputFile.txt", 1, [])

#LaunchOnCondor.ListToFile(LaunchOnCondor.GetListOfFiles('"','/store/user/farrell3/HSCPEDMUpdateTo4p3fb28Oct2011/RunA_*_172901_173243/HSCP_*.root','",'), FarmDirectory + "InputFile.txt")
#LaunchOnCondor.SendCMSJobs(FarmDirectory, "Data_RunA_172901_173243", "Merge_cfg.py", FarmDirectory + "InputFile.txt", 1, [])

#LaunchOnCondor.ListToFile(LaunchOnCondor.GetListOfFiles('"','/store/user/farrell3/HSCPEDMUpdateTo4p3fb28Oct2011/RunA_*_173244_173692/HSCP_*.root','",'), FarmDirectory + "InputFile.txt")
#LaunchOnCondor.SendCMSJobs(FarmDirectory, "Data_RunA_173244_173692", "Merge_cfg.py", FarmDirectory + "InputFile.txt", 1, [])

#LaunchOnCondor.ListToFile(LaunchOnCondor.GetListOfFiles('"','/store/user/farrell3/HSCPEDMUpdateTo4p3fb28Oct2011/RunA_*_175860_176099/HSCP_*.root','",'), FarmDirectory + "InputFile.txt")
#LaunchOnCondor.SendCMSJobs(FarmDirectory, "Data_RunA_175860_176099", "Merge_cfg.py", FarmDirectory + "InputFile.txt", 1, [])

#LaunchOnCondor.ListToFile(LaunchOnCondor.GetListOfFiles('"','/store/user/farrell3/HSCPEDMUpdateTo4p3fb28Oct2011/RunA_*_176100_176309/HSCP_*.root','",'), FarmDirectory + "InputFile.txt")
#LaunchOnCondor.SendCMSJobs(FarmDirectory, "Data_RunA_176100_176309", "Merge_cfg.py", FarmDirectory + "InputFile.txt", 1, [])

#LaunchOnCondor.ListToFile(LaunchOnCondor.GetListOfFiles('"','/store/user/farrell3/HSCPEDMUpdateTo4p3fb28Oct2011/RunA_*_176467_176800/HSCP_*.root','",'), FarmDirectory + "InputFile.txt")
#LaunchOnCondor.SendCMSJobs(FarmDirectory, "Data_RunA_176467_176800", "Merge_cfg.py", FarmDirectory + "InputFile.txt", 1, [])

#LaunchOnCondor.ListToFile(LaunchOnCondor.GetListOfFiles('"','/store/user/farrell3/HSCPEDMUpdateTo4p3fb28Oct2011/RunA_*_176801_177053/HSCP_*.root','",'), FarmDirectory + "InputFile.txt")
#LaunchOnCondor.SendCMSJobs(FarmDirectory, "Data_RunA_176801_177053", "Merge_cfg.py", FarmDirectory + "InputFile.txt", 1, [])

#LaunchOnCondor.ListToFile(LaunchOnCondor.GetListOfFiles('"','/store/user/farrell3/HSCPEDMUpdateTo4p3fb28Oct2011/RunA_*_177074_177783/HSCP_*.root','",'), FarmDirectory + "InputFile.txt")
#LaunchOnCondor.SendCMSJobs(FarmDirectory, "Data_RunA_177074_177783", "Merge_cfg.py", FarmDirectory + "InputFile.txt", 1, [])

#LaunchOnCondor.ListToFile(LaunchOnCondor.GetListOfFiles('"','/store/user/farrell3/HSCPEDMUpdateTo4p3fb28Oct2011/RunA_*_177788_178380/HSCP_*.root','",'), FarmDirectory + "InputFile.txt")
#LaunchOnCondor.SendCMSJobs(FarmDirectory, "Data_RunA_177788_178380", "Merge_cfg.py", FarmDirectory + "InputFile.txt", 1, [])

#LaunchOnCondor.ListToFile(LaunchOnCondor.GetListOfFiles('"','/store/user/farrell3/HSCPEDMUpdateTo4p3fb28Oct2011/RunA_*_178420_179411/HSCP_*.root','",'), FarmDirectory + "InputFile.txt")
#LaunchOnCondor.SendCMSJobs(FarmDirectory, "Data_RunA_178420_179411", "Merge_cfg.py", FarmDirectory + "InputFile.txt", 1, [])

LaunchOnCondor.ListToFile(LaunchOnCondor.GetListOfFiles('"','/store/user/farrell3/HSCPEDMUpdateTo4p3fb28Oct2011/RunA_*_179434_180252/HSCP_*.root','",'), FarmDirectory + "InputFile.txt")
LaunchOnCondor.SendCMSJobs(FarmDirectory, "Data_RunA_179434_180252", "Merge_cfg.py", FarmDirectory + "InputFile.txt", 1, [])

os.system("rm " +  FarmDirectory + "InputFile.txt")
