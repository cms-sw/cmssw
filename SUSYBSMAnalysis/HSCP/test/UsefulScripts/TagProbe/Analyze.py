#!/usr/bin/env python

import urllib
import string
import os
import sys
import LaunchOnCondor

Jobs = [
#["HSCPTagProbeTree", "TkProbe_TkCut_Eta"],
#["HSCPTagProbeTree", "TkProbe_TkCut_Pt"],
#["HSCPTagProbeTree", "TkProbe_TkCut_PV"],
#["HSCPTagProbeTree", "TkProbe_TkTOFCut_Eta"],
#["HSCPTagProbeTree", "TkProbe_TkTOFCut_Pt"],
#["HSCPTagProbeTree", "TkProbe_TkTOFCut_PV"],

["HSCPTagProbeTree", "TkProbe_SACut_Eta"],
["HSCPTagProbeTree", "TkProbe_StationCut_Eta"],
["HSCPTagProbeTree", "TkProbe_DxyCut_Eta"],
["HSCPTagProbeTree", "TkProbe_SegSepCut_Eta"],
["HSCPTagProbeTree", "TkProbe_DzCut_Eta"],
["HSCPTagProbeTree", "TkProbe_MuOnlyCut_Eta"],
#["HSCPTagProbeTree", "TkProbe_MuOnlyCut_Pt"],
#["HSCPTagProbeTree", "TkProbe_MuOnlyCut_PV"],
["HSCPTagProbeTree", "TkProbe_NdofCut_Eta"],
#["HSCPTagProbeTree", "TkProbe_NdofCut_Pt"],
#["HSCPTagProbeTree", "TkProbe_NofCut_PV"],

#["HSCPTagProbeTreeMC", "TkProbe_TkCut_Eta_MC"],
#["HSCPTagProbeTreeMC", "TkProbe_TkCut_Pt_MC"],
#["HSCPTagProbeTreeMC", "TkProbe_TkCut_PV_MC"],
#["HSCPTagProbeTreeMC", "TkProbe_TkTOFCut_Eta_MC"],
#["HSCPTagProbeTreeMC", "TkProbe_TkTOFCut_Pt_MC"],
#["HSCPTagProbeTreeMC", "TkProbe_TkTOFCut_PV_MC"],

["HSCPTagProbeTreeMC", "TkProbe_SACut_Eta_MC"],
["HSCPTagProbeTreeMC", "TkProbe_StationCut_Eta_MC"],
["HSCPTagProbeTreeMC", "TkProbe_DxyCut_Eta_MC"],
["HSCPTagProbeTreeMC", "TkProbe_SegSepCut_Eta_MC"],
["HSCPTagProbeTreeMC", "TkProbe_DzCut_Eta_MC"],
["HSCPTagProbeTreeMC", "TkProbe_MuOnlyCut_Eta_MC"],
#["HSCPTagProbeTreeMC", "TkProbe_MuOnlyCut_Pt_MC"],
#["HSCPTagProbeTreeMC", "TkProbe_MuOnlyCut_PV_MC"],
["HSCPTagProbeTreeMC", "TkProbe_NdofCut_Eta_MC"],
#["HSCPTagProbeTreeMC", "TkProbe_NdofCut_Pt_MC"],
#["HSCPTagProbeTreeMC", "TkProbe_NdofCut_PV_MC"],

#["HSCPTagProbeTree", "SAProbe_TkCut_Eta"],
#["HSCPTagProbeTree", "SAProbe_TkCut_Pt"],
#["HSCPTagProbeTree", "SAProbe_TkCut_PV"],
#["HSCPTagProbeTree", "SAProbe_TkTOFCut_Eta"],
#["HSCPTagProbeTree", "SAProbe_TkTOFCut_Pt"],
#["HSCPTagProbeTree", "SAProbe_TkTOFCut_PV"],
#["HSCPTagProbeTreeMC", "SAProbe_TkCut_Eta_MC"],
#["HSCPTagProbeTreeMC", "SAProbe_TkCut_Pt_MC"],
#["HSCPTagProbeTreeMC", "SAProbe_TkCut_PV_MC"],
#["HSCPTagProbeTreeMC", "SAProbe_TkTOFCut_Eta_MC"],
#["HSCPTagProbeTreeMC", "SAProbe_TkTOFCut_Pt_MC"],
#["HSCPTagProbeTreeMC", "SAProbe_TkTOFCut_PV_MC"],
]

FarmDirectory = "ANALYSE"
for i in range(len(Jobs)):
        j=Jobs[i]
        #Dir = "/uscms_data/d2/farrell3/WorkArea/14Aug2012/CMSSW_5_3_3/src/PhysicsTools/TagAndProbe/test/"
	#LaunchOnCondor.ListToFile(LaunchOnCondor.GetListOfFiles('"',Dir+j[0]+'.root','"'), FarmDirectory + "InputFile.txt")

        Dir = "/uscms_data/d2/farrell3/WorkArea/14Aug2012/CMSSW_5_3_3/src/PhysicsTools/TagAndProbe/test/TagProbeProducerRoot/"
        LaunchOnCondor.ListToFile(LaunchOnCondor.GetListOfFiles('"file:',Dir+j[0]+'.root','"'), FarmDirectory + "InputFile.txt")

	LaunchOnCondor.SendCMSJobs(FarmDirectory, j[1], "HSCPTagProbeTreeAnalyzer.py", FarmDirectory + "InputFile.txt", 1, ['XXX_SAVEPATH_XXX','file:/storage/data/cms/users/quertenmont/HSCP/CMSSW_4_2_8/12_08_16/'])
        fo = open("Run" + str(i) + ".sh", "w")
        fo.write("cmsRun ANALYSE/inputs/0000_" + j[1] + "_cfg.py > ANALYSE/inputs/0000_" + j[1] + "_cfg.out")
        fo.close()
        os.system("chmod 755 Run" + str(i) + ".sh")
        os.system("rm " +  FarmDirectory + "InputFile.txt")
