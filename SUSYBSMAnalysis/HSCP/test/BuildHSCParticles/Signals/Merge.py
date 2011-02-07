#!/usr/bin/env python

import urllib
import string
import os
import sys
import LaunchOnCondor



Jobs = ["DCStau121","DCStau182","DCStau242","DCStau302","Gluino200","Gluino200N","Gluino300","Gluino300N","Gluino400","Gluino400N","Gluino500","Gluino500N","Gluino600","Gluino600N","Gluino900","Gluino900N","PPStau100","PPStau126","PPStau156","PPStau200","PPStau247","PPStau308","Stop130","Stop130N","Stop200","Stop200N","Stop300","Stop300N","Stop500","Stop500N","Stop800","Stop800N","mGMSBStau100","mGMSBStau126","mGMSBStau156","mGMSBStau200","mGMSBStau247","mGMSBStau308"]

#Jobs = ["Stau100","Stau126", "Stau156", "Stau200", "Stau247", "Stau308", "Stop130", "Stop200", "Stop300", "Stop500", "Stop800", "Gluino200", "Gluino300", "Gluino400", "Gluino500", "Gluino600", "Gluino900", "MGStop130", "MGStop200", "MGStop300", "MGStop500", "MGStop800", "Stop130Neutral", "Stop200Neutral", "Stop300Neutral", "Stop500Neutral", "Stop800Neutral", "Gluino200Neutral", "Gluino300Neutral", "Gluino400Neutral", "Gluino500Neutral", "Gluino600Neutral", "Gluino900Neutral"]
FarmDirectory = "MERGE"
for JobName in Jobs:
	InputFiles    = LaunchOnCondor.GetListOfFiles('"file:','/storage/data/cms/store/user/quertenmont/11_01_28_HSCP40pb/FWLite_Sign/' + JobName + '/HSCP_*.root','",');
	LaunchOnCondor.SendCMSMergeJob(FarmDirectory, JobName, InputFiles, '"XXX_OUTPUT_XXX.root"', '"keep *"')
