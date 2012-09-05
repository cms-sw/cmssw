#!/usr/bin/env python

import urllib
import string
import os
import sys
import LaunchOnCondor
import glob

print 'OPTIMIZATION'
FarmDirectory = "FARM"
JobName = "EFFICIENCTY"
LaunchOnCondor.Jobs_RunHere = 1
LaunchOnCondor.SendCluster_Create(FarmDirectory, JobName)
LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/TriggerEfficiency.C", '"ANALYSE"'])
LaunchOnCondor.SendCluster_Submit()
