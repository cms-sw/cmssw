#!/usr/bin/env python

import urllib
import string
import os
import sys
import LaunchOnCondor

Jobs = ["DCStau121","DCStau182","DCStau242","DCStau302","Gluino200","Gluino200N","Gluino300","Gluino300N","Gluino400","Gluino400N","Gluino500","Gluino500N","Gluino600","Gluino600N","Gluino900","Gluino900N","PPStau100","PPStau126","PPStau156","PPStau200","PPStau247","PPStau308","Stop130","Stop130N","Stop200","Stop200N","Stop300","Stop300N","Stop500","Stop500N","Stop800","Stop800N","GMStau100","GMStau126","GMStau156","GMStau200","GMStau247","GMStau308"]

FarmDirectory = "MERGE"
for JobName in Jobs:
	LaunchOnCondor.ListToFile(LaunchOnCondor.GetListOfFiles('"file:','/storage/data/cms/store/user/quertenmont/11_04_02_HSCP2011/FWLite_Sign/'+ JobName + '/HSCP_*.root','",'), FarmDirectory + "InputFile.txt") 
	LaunchOnCondor.SendCMSJobs(FarmDirectory, JobName, "Merge_cfg.py", FarmDirectory + "InputFile.txt", 1, [])

os.system("rm " +  FarmDirectory + "InputFile.txt")
