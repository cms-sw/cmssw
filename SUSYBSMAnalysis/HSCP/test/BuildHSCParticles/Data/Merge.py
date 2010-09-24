#!/usr/bin/env python

import urllib
import string
import os
import sys
import LaunchOnCondor

FarmDirectory = "MERGE"

InputFiles    = LaunchOnCondor.GetListOfFiles('"file:','/storage/data/cms/store/user/quertenmont/10_09_11_HSCP/FWLite_Data2/Run131511_135802_SD*/*.root','",');
LaunchOnCondor.SendCMSMergeJob(FarmDirectory, "Data_131511_135802", InputFiles, '"XXX_OUTPUT_XXX.root"', '"keep *"')

InputFiles    = LaunchOnCondor.GetListOfFiles('"file:','/storage/data/cms/store/user/quertenmont/10_09_11_HSCP/FWLite_Data2/Run135821_137433_SD*/*.root','",');
LaunchOnCondor.SendCMSMergeJob(FarmDirectory, "Data_135821_137433", InputFiles, '"XXX_OUTPUT_XXX.root"', '"keep *"')

#InputFiles    = LaunchOnCondor.GetListOfFiles('"file:','/storage/data/cms/store/user/quertenmont/10_09_11_HSCP/FWLite_Data2/Run137436_141887_SD*/*.root','",');
#LaunchOnCondor.SendCMSMergeJob(FarmDirectory, "Data_137436_to_141887", InputFiles, '"XXX_OUTPUT_XXX.root"', '"keep *"')

#InputFiles    = LaunchOnCondor.GetListOfFiles('"file:','/storage/data/cms/store/user/quertenmont/10_09_11_HSCP/FWLite_Data2/Run141888_142000_SD*/*.root','",');
#LaunchOnCondor.SendCMSMergeJob(FarmDirectory, "Data_141888_to_142000", InputFiles, '"XXX_OUTPUT_XXX.root"', '"keep *"')

#InputFiles    = LaunchOnCondor.GetListOfFiles('"file:','/storage/data/cms/store/user/quertenmont/10_09_11_HSCP/FWLite_Data2/Run142000_143000_SD*/*.root','",');
#LaunchOnCondor.SendCMSMergeJob(FarmDirectory, "Data_142000_to_143000", InputFiles, '"XXX_OUTPUT_XXX.root"', '"keep *"')

#InputFiles    = LaunchOnCondor.GetListOfFiles('"file:','/storage/data/cms/store/user/quertenmont/10_09_11_HSCP/FWLite_Data2/Run143000_144000_SD*/*.root','",');
#LaunchOnCondor.SendCMSMergeJob(FarmDirectory, "Data_143000_to_144000", InputFiles, '"XXX_OUTPUT_XXX.root"', '"keep *"')

#InputFiles    = LaunchOnCondor.GetListOfFiles('"file:','/storage/data/cms/store/user/quertenmont/10_09_11_HSCP/FWLite_Data2/Run144000_144114_SDJetMET/*.root','",');
#InputFiles   += LaunchOnCondor.GetListOfFiles('"file:','/storage/data/cms/store/user/quertenmont/10_09_11_HSCP/FWLite_Data2/Run144400_144114_SDMu/*.root','",');
#LaunchOnCondor.SendCMSMergeJob(FarmDirectory, "Data_144000_to_144114", InputFiles, '"XXX_OUTPUT_XXX.root"', '"keep *"')


