#!/usr/bin/env python

import urllib
import string
import os
import sys
import LaunchOnCondor

Jobs = ["GMStau100", "GMStau100BX1", "GMStau126", "GMStau126BX1", "GMStau156", "GMStau156BX1", "GMStau200", "GMStau200BX1", "GMStau247", "GMStau247BX1", "GMStau308", "GMStau308BX1", "GMStau370", "GMStau370BX1", "GMStau432", "GMStau432BX1", "GMStau494", "GMStau494BX1", "Gluino300", "Gluino300N", "Gluino300NBX1", "Gluino300BX1", "Gluino400", "Gluino400N", "Gluino400NBX1", "Gluino400BX1", "Gluino500", "Gluino500N", "Gluino500NBX1", "Gluino500BX1", "Gluino600", "Gluino600N", "Gluino600NBX1", "Gluino600Z", "Gluino600ZBX1", "Gluino600BX1", "Gluino700", "Gluino700N", "Gluino700NBX1", "Gluino700Z", "Gluino700ZBX1", "Gluino700BX1", "Gluino800", "Gluino800N", "Gluino800NBX1", "Gluino800Z", "Gluino800ZBX1", "Gluino800BX1", "Gluino900", "Gluino900N", "Gluino900NBX1", "Gluino900BX1", "Gluino1000", "Gluino1000N", "Gluino1000NBX1", "Gluino1000BX1", "Gluino1100", "Gluino1100N", "Gluino1100NBX1", "Gluino1100BX1", "Gluino1200", "Gluino1200N", "Gluino1200NBX1", "Gluino1200BX1", "Stop130", "Stop130N", "Stop130NBX1", "Stop130BX1", "Stop200", "Stop200N", "Stop200NBX1", "Stop200BX1", "Stop300", "Stop300N", "Stop300NBX1", "Stop300Z", "Stop300ZBX1", "Stop300BX1", "Stop400", "Stop400N", "Stop400NBX1", "Stop400Z", "Stop400ZBX1", "Stop400BX1", "Stop500", "Stop500N", "Stop500NBX1", "Stop500Z", "Stop500ZBX1", "Stop500BX1", "Stop600", "Stop600N", "Stop600NBX1", "Stop600BX1", "Stop700", "Stop700N", "Stop700NBX1", "Stop700BX1", "Stop800", "Stop800N", "Stop800NBX1", "Stop800BX1"]

FarmDirectory = "MERGE"
for JobName in Jobs:
	LaunchOnCondor.ListToFile(LaunchOnCondor.GetListOfFiles('"file:','/storage/data/cms/store/user/quertenmont/11_07_15_HSCP2011/FWLite_Sign/'+ JobName + '/HSCP_*.root','",'), FarmDirectory + "InputFile.txt") 
	LaunchOnCondor.SendCMSJobs(FarmDirectory, JobName, "Merge_cfg.py", FarmDirectory + "InputFile.txt", 1, [])

os.system("rm " +  FarmDirectory + "InputFile.txt")
