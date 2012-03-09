#!/usr/bin/env python

import urllib
import string
import os
import sys
import LaunchOnCondor

FarmDirectory = "MERGE"
InputFiles    = LaunchOnCondor.GetListOfFiles('"file:','/storage/data/cms/store/user/quertenmont/10_09_11_HSCP/FWLite_MC/MB/*.root','",');
LaunchOnCondor.SendCMSMergeJob(FarmDirectory, "MC_MB", InputFiles, '"XXX_OUTPUT_XXX.root"', '"keep *"')

InputFiles    = LaunchOnCondor.GetListOfFiles('"file:','/storage/data/cms/store/user/quertenmont/10_09_11_HSCP/FWLite_MC/QCDPT30/*.root','",');
LaunchOnCondor.SendCMSMergeJob(FarmDirectory, "MC_QCD30", InputFiles, '"XXX_OUTPUT_XXX.root"', '"keep *"')

InputFiles    = LaunchOnCondor.GetListOfFiles('"file:','/storage/data/cms/store/user/quertenmont/10_09_11_HSCP/FWLite_MC/QCDPT80/*.root','",');
LaunchOnCondor.SendCMSMergeJob(FarmDirectory, "MC_QCD80", InputFiles, '"XXX_OUTPUT_XXX.root"', '"keep *"')

InputFiles    = LaunchOnCondor.GetListOfFiles('"file:','/storage/data/cms/store/user/quertenmont/10_09_11_HSCP/FWLite_MC/PPMUX/*.root','",');
LaunchOnCondor.SendCMSMergeJob(FarmDirectory, "MC_PPMUX", InputFiles, '"XXX_OUTPUT_XXX.root"', '"keep *"')

