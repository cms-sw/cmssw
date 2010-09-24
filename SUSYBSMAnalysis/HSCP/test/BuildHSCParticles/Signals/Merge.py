#!/usr/bin/env python

import urllib
import string
import os
import sys
import LaunchOnCondor

Jobs = ["Stau100","Stau126", "Stau156", "Stau200", "Stau247", "Stau308", "Stop130", "Stop200", "Stop300", "Stop500", "Stop800", "Gluino200", "Gluino300", "Gluino400", "Gluino500", "Gluino600", "Gluino900", "MGStop130", "MGStop200", "MGStop300", "MGStop500", "MGStop800", "Stop130Neutral", "Stop200Neutral", "Stop300Neutral", "Stop500Neutral", "Stop800Neutral", "Gluino200Neutral", "Gluino300Neutral", "Gluino400Neutral", "Gluino500Neutral", "Gluino600Neutral", "Gluino900Neutral"]
FarmDirectory = "MERGE"
for JobName in Jobs:
	InputFiles    = LaunchOnCondor.GetListOfFiles('"file:','/storage/data/cms/store/user/quertenmont/10_09_11_HSCP/FWLite_SignReReco/' + JobName + '/HSCP_*.root','",');
	LaunchOnCondor.SendCMSMergeJob(FarmDirectory, JobName, InputFiles, '"XXX_OUTPUT_XXX.root"', '"keep *"')
