#!/usr/bin/env python

import urllib
import string
import os
import sys
import LaunchOnCondor

Jobs = [
"190645_190999",
"191000_191999",
"192000_192999",
"193000_193621",
"193622_193999",
"194000_194999",
"195000_195999",
"196000_196531",
"197000_197999",
"198000_198345",
"198488_198919",
"198920_198999",
"199000_199999",
"200000_200532",
"200533_202016",
"200533_202016",
"202017_203002",
"MC_8TeV_DYToMuMu"
]

FarmDirectory = "FARM"
for j in Jobs:
	LaunchOnCondor.ListToFile(LaunchOnCondor.GetListOfFiles('"dcache:','/pnfs/cms/WAX/11/store/user/lpchscp/2012HSCPEDMFiles/*'+j+'*.root','",'), FarmDirectory + "InputFile.txt")
	LaunchOnCondor.SendCMSJobs(FarmDirectory, j, "HSCPTagProbeTreeProducer.py", FarmDirectory + "InputFile.txt", 1, ['XXX_SAVEPATH_XXX','file:/storage/data/cms/users/quertenmont/HSCP/CMSSW_4_2_8/12_08_16/'])
os.system("rm " +  FarmDirectory + "InputFile.txt")
