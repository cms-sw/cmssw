#!/usr/bin/env python

import urllib
import string
import os
import sys
import LaunchOnCondor

FarmDirectory = "MERGE"

InputFiles    = LaunchOnCondor.GetListOfFiles('"file:','/storage/data/cms/store/user/quertenmont/11_01_28_HSCP40pb/FWLite_Data/Run135821_141887_SD*/*.root','",');
LaunchOnCondor.SendCMSMergeJob(FarmDirectory, "Data_135821_141887", InputFiles, '"XXX_OUTPUT_XXX.root"', '"keep *"')

InputFiles    = LaunchOnCondor.GetListOfFiles('"file:','/storage/data/cms/store/user/quertenmont/11_01_28_HSCP40pb/FWLite_Data/Run141888_144114_SD*/*.root','",');
LaunchOnCondor.SendCMSMergeJob(FarmDirectory, "Data_141888_144114", InputFiles, '"XXX_OUTPUT_XXX.root"', '"keep *"')

InputFiles    = LaunchOnCondor.GetListOfFiles('"file:','/storage/data/cms/store/user/quertenmont/11_01_28_HSCP40pb/FWLite_Data/Run146240_147000_SD*/*.root','",');
LaunchOnCondor.SendCMSMergeJob(FarmDirectory, "Data_146240_147000", InputFiles, '"XXX_OUTPUT_XXX.root"', '"keep *"')

InputFiles    = LaunchOnCondor.GetListOfFiles('"file:','/storage/data/cms/store/user/quertenmont/11_01_28_HSCP40pb/FWLite_Data/Run147001_148000_SD*/*.root','",');
LaunchOnCondor.SendCMSMergeJob(FarmDirectory, "Data_147001_148000", InputFiles, '"XXX_OUTPUT_XXX.root"', '"keep *"')

InputFiles    = LaunchOnCondor.GetListOfFiles('"file:','/storage/data/cms/store/user/quertenmont/11_01_28_HSCP40pb/FWLite_Data/Run148001_149000_SD*/*.root','",');
LaunchOnCondor.SendCMSMergeJob(FarmDirectory, "Data_148001_149000", InputFiles, '"XXX_OUTPUT_XXX.root"', '"keep *"')

InputFiles    = LaunchOnCondor.GetListOfFiles('"file:','/storage/data/cms/store/user/quertenmont/11_01_28_HSCP40pb/FWLite_Data/Run149001_149711_SD*/*.root','",');
LaunchOnCondor.SendCMSMergeJob(FarmDirectory, "Data_149001_149711", InputFiles, '"XXX_OUTPUT_XXX.root"', '"keep *"')
