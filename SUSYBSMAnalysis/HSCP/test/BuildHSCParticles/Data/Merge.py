#!/usr/bin/env python

import urllib
import string
import os
import sys
import LaunchOnCondor

FarmDirectory = "MERGE2"

LaunchOnCondor.ListToFile(LaunchOnCondor.GetListOfFiles('"file:','/storage/data/cms/store/user/quertenmont/11_02_15_HSCP40pb/FWLite_Data/Run135821_141887_SD*/*.root','",'), FarmDirectory + "InputFile.txt") 
LaunchOnCondor.SendCMSJobs(FarmDirectory, "Data_135821_141887", "Merge_cfg.py", FarmDirectory + "InputFile.txt", 1, [])

LaunchOnCondor.ListToFile(LaunchOnCondor.GetListOfFiles('"file:','/storage/data/cms/store/user/quertenmont/11_02_15_HSCP40pb/FWLite_Data/Run141888_144114_SD*/*.root','",'), FarmDirectory + "InputFile.txt") 
LaunchOnCondor.SendCMSJobs(FarmDirectory, "Data_141888_144114", "Merge_cfg.py", FarmDirectory + "InputFile.txt", 1, [])

LaunchOnCondor.ListToFile(LaunchOnCondor.GetListOfFiles('"file:','/storage/data/cms/store/user/quertenmont/11_02_15_HSCP40pb/FWLite_Data/Run146240_147000_SD*/*.root','",'), FarmDirectory + "InputFile.txt") 
LaunchOnCondor.SendCMSJobs(FarmDirectory, "Data_146240_147000", "Merge_cfg.py", FarmDirectory + "InputFile.txt", 1, [])

LaunchOnCondor.ListToFile(LaunchOnCondor.GetListOfFiles('"file:','/storage/data/cms/store/user/quertenmont/11_02_15_HSCP40pb/FWLite_Data/Run147001_148000_SD*/*.root','",'), FarmDirectory + "InputFile.txt") 
LaunchOnCondor.SendCMSJobs(FarmDirectory, "Data_147001_148000", "Merge_cfg.py", FarmDirectory + "InputFile.txt", 1, [])

LaunchOnCondor.ListToFile(LaunchOnCondor.GetListOfFiles('"file:','/storage/data/cms/store/user/quertenmont/11_02_15_HSCP40pb/FWLite_Data/Run148001_148500_SD*/*.root','",'), FarmDirectory + "InputFile.txt") 
LaunchOnCondor.SendCMSJobs(FarmDirectory, "Data_148001_148500", "Merge_cfg.py", FarmDirectory + "InputFile.txt", 1, [])

LaunchOnCondor.ListToFile(LaunchOnCondor.GetListOfFiles('"file:','/storage/data/cms/store/user/quertenmont/11_02_15_HSCP40pb/FWLite_Data/Run148501_149000_SD*/*.root','",'), FarmDirectory + "InputFile.txt")
LaunchOnCondor.SendCMSJobs(FarmDirectory, "Data_148501_149000", "Merge_cfg.py", FarmDirectory + "InputFile.txt", 1, [])

LaunchOnCondor.ListToFile(LaunchOnCondor.GetListOfFiles('"file:','/storage/data/cms/store/user/quertenmont/11_02_15_HSCP40pb/FWLite_Data/Run149001_149250_SD*/*.root','",'), FarmDirectory + "InputFile.txt") 
LaunchOnCondor.SendCMSJobs(FarmDirectory, "Data_149001_149250", "Merge_cfg.py", FarmDirectory + "InputFile.txt", 1, [])

LaunchOnCondor.ListToFile(LaunchOnCondor.GetListOfFiles('"file:','/storage/data/cms/store/user/quertenmont/11_02_15_HSCP40pb/FWLite_Data/Run149251_149711_SD*/*.root','",'), FarmDirectory + "InputFile.txt")
LaunchOnCondor.SendCMSJobs(FarmDirectory, "Data_149251_149711", "Merge_cfg.py", FarmDirectory + "InputFile.txt", 1, [])


os.system("rm " +  FarmDirectory + "InputFile.txt")
