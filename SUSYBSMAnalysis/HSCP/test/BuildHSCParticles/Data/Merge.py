#!/usr/bin/env python

import urllib
import string
import os
import sys
import LaunchOnCondor

Jobs = [
#["RunA_*_190645_190999","RunA_190645_190999"],
#["RunA_*_191000_191999","RunA_191000_191999"],
#["RunA_*_192000_192999","RunA_192000_192999"],
#["RunA_*_193000_193621","RunA_193000_193621"],
#["RunB_*_193622_193999","RunB_193622_193999"],
#["RunB_*_194000_194999","RunB_194000_194999"],
#["RunB_*_195000_195999","RunB_195000_195999"],
#["RunB_*_196000_196531","RunB_196000_196531"],
#["RunC_*_197000_197999","RunC_196532_197999"],
#["RunC_*_198000_198345","RunC_198000_198345"],
#["RunC_*_198488_198919","RunC_198488_198919"],
#["RunC_*_198920_198999","RunC_198920_198999"],
#["RunC_*_199000_199999","RunC_199000_199999"],
#["RunC_*_200000_200532","RunC_200000_200532"],
#["RunC_*_200533_202016","RunC_200533_202016"],
#["Run2012C_*_202017_203002","RunC_202017_203002"],

#["RunD_*_203003_203300","RunD_203003_203300"],
#["RunD_*_203301_203600","RunD_203301_203600"],
#["RunD_*_203601_203900","RunD_203601_203900"],
#["RunD_*_203901_204200","RunD_203901_204200"],
#["RunD_*_204201_204500","RunD_204201_204500"],
#["RunD_*_204501_204800","RunD_204501_204800"],
#["RunD_*_204801_205100","RunD_204801_205100"],
#["RunD_*_205101_205400","RunD_205101_205400"],
#["RunD_*_205401_205700","RunD_205401_205700"],
#["RunD_*_205701_206000","RunD_205701_206000"],
#["RunD_*_206001_206300","RunD_206001_206300"],
#["RunD_*_206301_206600","RunD_206301_206600"],
#["RunD_*_206601_206900","RunD_206601_206900"],
#["RunD_*_206901_207200","RunD_206901_207200"],
#["RunD_*_207201_207500","RunD_207201_207500"],
#["RunD_*_207501_207800","RunD_207501_207800"],
#["RunD_*_207801_208100","RunD_207801_208100"],
#["RunD_*_208101_208357","RunD_208101_208357"],
["RunD_*_208358_208686","RunD_208358_208686"]
]

FarmDirectory = "MERGE"
for j in Jobs:
	LaunchOnCondor.ListToFile(LaunchOnCondor.GetListOfFiles('"dcache:','/pnfs/cms/WAX/11/store/user/farrell3/HSCPEDMUpdateData2012_30Nov2012/'+j[0]+'/HSCP_*.root','",'), FarmDirectory + "InputFile.txt")
	#LaunchOnCondor.SendCMSJobs(FarmDirectory, j[1], "Merge_cfg.py", FarmDirectory + "InputFile.txt", 1, ['XXX_SAVEPATH_XXX','file:/storage/data/cms/users/quertenmont/HSCP/CMSSW_4_2_8/12_08_16/'])
	LaunchOnCondor.SendCMSJobs(FarmDirectory, j[1], "Merge_cfg.py", FarmDirectory + "InputFile.txt", 1, ['XXX_SAVEPATH_XXX','/uscmst1b_scratch/lpc1/3DayLifetime/farrell/2012Data_04Sep2012'])
os.system("rm " +  FarmDirectory + "InputFile.txt")
