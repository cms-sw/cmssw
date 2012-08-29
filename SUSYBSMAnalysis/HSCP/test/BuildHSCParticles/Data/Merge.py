#!/usr/bin/env python

import urllib
import string
import os
import sys
import LaunchOnCondor

Jobs = [
#2011
["FWLite_Data/RunA_*_160329_164236","RunA_160329_164236"],
["FWLite_Data/RunA_*_165071_166033","RunA_165071_166033"],
["FWLite_Data/RunA_*_166034_168437","RunA_166034_168437"],
["FWLite_Data/RunA_*_170053_172619","RunA_170053_172619"],
["FWLite_Data/RunA_*_172791_175770","RunA_172791_175770"],
["FWLite_Data/RunB_*_175831_177999","RunB_175831_177999"],
["FWLite_Data/RunB_*_178000_178999","RunB_178000_178999"],
["FWLite_Data/RunB_*_179000_180296","RunB_179000_180296"],
#2012
["FWLite_Data12/RunA_*_190645_190999","RunA_190645_190999"],
["FWLite_Data12/RunA_*_191000_191999","RunA_191000_191999"],
["FWLite_Data12/RunA_*_192000_192999","RunA_192000_192999"],
["FWLite_Data12/RunA_*_193000_193621","RunA_193000_193621"],
["FWLite_Data12/RunB_*_193622_193999","RunB_193622_193999"],
["FWLite_Data12/RunB_*_194000_194999","RunB_194000_194999"],
["FWLite_Data12/RunB_*_195000_195999","RunB_195000_195999"],
["FWLite_Data12/RunB_*_196000_196531","RunB_196000_196531"],
["FWLite_Data12/RunC_*_197000_197999","RunC_196532_197999"],
["FWLite_Data12/RunC_*_198000_198345","RunC_198000_198345"],
["FWLite_Data12/RunC_*_198488_198919","RunC_198488_198919"],
["FWLite_Data12/RunC_*_198920_198999","RunC_198920_198999"],
["FWLite_Data12/RunC_*_199000_199999","RunC_199000_199999"],
["FWLite_Data12/RunC_*_200000_200532","RunC_200000_200532"],
]

FarmDirectory = "MERGE"
for j in Jobs:
	LaunchOnCondor.ListToFile(LaunchOnCondor.GetListOfFiles('"file:','/storage/data/cms/store/user/quertenmont/12_08_16_HSCP_EDM2011/'+j[0]+'/HSCP_*.root','",'), FarmDirectory + "InputFile.txt")
	LaunchOnCondor.SendCMSJobs(FarmDirectory, j[1], "Merge_cfg.py", FarmDirectory + "InputFile.txt", 1, ['XXX_SAVEPATH_XXX','file:/storage/data/cms/users/quertenmont/HSCP/CMSSW_4_2_8/12_08_16/'])
os.system("rm " +  FarmDirectory + "InputFile.txt")
