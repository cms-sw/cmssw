#!/usr/bin/env python

import urllib
import string
import os
import sys
import LaunchOnCondor
import glob


def ScanWP(Arguments):
	WP_Pt=0
	while WP_Pt>=-5:
        	WP_I=0
	        while WP_I>=-5:
        		Arguments[3] = WP_Pt
		        Arguments[4] = WP_I
                        LaunchOnCondor.SendCluster_Push(Arguments)
                	WP_I=WP_I-0.5
	        WP_Pt=WP_Pt-0.5

if len(sys.argv)==1:
	print "Please pass in argument a number between 0 and 2"
	sys.exit()

if sys.argv[1]=='0':
        print 'ANALYSIS'
        FarmDirectory = "FARM"
        JobName = "HscpAnalysis"
        LaunchOnCondor.SendCluster_Create(FarmDirectory, JobName)	
        ScanWP(["FWLITE", "Analysis_Step2345.C", '"ANALYSE"', "WP_Pt", "WP_I", 2, 11, 3, 1])
        ScanWP(["FWLITE", "Analysis_Step2345.C", '"ANALYSE"', "WP_Pt", "WP_I", 2, 11, 3, 0])
#	ScanWP(["FWLITE", "Analysis_Step2345.C", '"ANALYSE"', "WP_Pt", "WP_I", 2,  8, 0, 1])
#	ScanWP(["FWLITE", "Analysis_Step2345.C", '"ANALYSE"', "WP_Pt", "WP_I", 2,  8, 0, 0])
#	ScanWP(["FWLITE", "Analysis_Step2345.C", '"ANALYSE"', "WP_Pt", "WP_I", 0, 11, 3, 1])
# 	ScanWP(["FWLITE", "Analysis_Step2345.C", '"ANALYSE"', "WP_Pt", "WP_I", 0, 11, 3, 0])
#	ScanWP(["FWLITE", "Analysis_Step2345.C", '"ANALYSE"', "WP_Pt", "WP_I", 0, 8, 0, 1])
#	ScanWP(["FWLITE", "Analysis_Step2345.C", '"ANALYSE"', "WP_Pt", "WP_I", 0, 8, 0, 0])
#	ScanWP(["FWLITE", "Analysis_Step2345.C", '"ANALYSE"', "WP_Pt", "WP_I", 0, 3, 3, 1])
# 	ScanWP(["FWLITE", "Analysis_Step2345.C", '"ANALYSE"', "WP_Pt", "WP_I", 0, 3, 3, 0])
#	ScanWP(["FWLITE", "Analysis_Step2345.C", '"ANALYSE"', "WP_Pt", "WP_I", 1, 11, 3, 1])
#	ScanWP(["FWLITE", "Analysis_Step2345.C", '"ANALYSE"', "WP_Pt", "WP_I", 1, 11, 3, 0])
#	ScanWP(["FWLITE", "Analysis_Step2345.C", '"ANALYSE"', "WP_Pt", "WP_I", 1, 8, 0, 1])
#	ScanWP(["FWLITE", "Analysis_Step2345.C", '"ANALYSE"', "WP_Pt", "WP_I", 1, 8, 0, 0])
        LaunchOnCondor.SendCluster_Submit()
elif sys.argv[1]=='1':
	print 'MAP MERGING'
        FarmDirectory = "FARM"
        JobName       = "HscpMerge"
        LaunchOnCondor.SendCluster_Create(FarmDirectory, JobName)
        LaunchOnCondor.SendCluster_Push  (["FWLITE", "Analysis_Step2345.C", '"MERGE_MAP"', -1, -1, 2, 11, 3, 1])
        LaunchOnCondor.SendCluster_Push  (["FWLITE", "Analysis_Step2345.C", '"MERGE_MAP"', -1, -1, 2, 11, 3, 0])
        LaunchOnCondor.SendCluster_Push  (["FWLITE", "Analysis_Step2345.C", '"MERGE_MAP"', -1, -1, 2,  8, 0, 1])
        LaunchOnCondor.SendCluster_Push  (["FWLITE", "Analysis_Step2345.C", '"MERGE_MAP"', -1, -1, 2,  8, 0, 0])
        LaunchOnCondor.SendCluster_Push  (["FWLITE", "Analysis_Step2345.C", '"MERGE_MAP"', -1, -1, 0, 11, 3, 1])
        LaunchOnCondor.SendCluster_Push  (["FWLITE", "Analysis_Step2345.C", '"MERGE_MAP"', -1, -1, 0, 11, 3, 0])
        LaunchOnCondor.SendCluster_Push  (["FWLITE", "Analysis_Step2345.C", '"MERGE_MAP"', -1, -1, 0,  8, 0, 1])
        LaunchOnCondor.SendCluster_Push  (["FWLITE", "Analysis_Step2345.C", '"MERGE_MAP"', -1, -1, 0,  8, 0, 0])
        LaunchOnCondor.SendCluster_Push  (["FWLITE", "Analysis_Step2345.C", '"MERGE_MAP"', -1, -1, 0,  3, 3, 1])
        LaunchOnCondor.SendCluster_Push  (["FWLITE", "Analysis_Step2345.C", '"MERGE_MAP"', -1, -1, 0,  3, 3, 0])
        LaunchOnCondor.SendCluster_Push  (["FWLITE", "Analysis_Step2345.C", '"MERGE_MAP"', -1, -1, 1, 11, 3, 1])
        LaunchOnCondor.SendCluster_Push  (["FWLITE", "Analysis_Step2345.C", '"MERGE_MAP"', -1, -1, 1, 11, 3, 0])
        LaunchOnCondor.SendCluster_Push  (["FWLITE", "Analysis_Step2345.C", '"MERGE_MAP"', -1, -1, 1,  8, 0, 1])
        LaunchOnCondor.SendCluster_Push  (["FWLITE", "Analysis_Step2345.C", '"MERGE_MAP"', -1, -1, 1,  8, 0, 0])
        LaunchOnCondor.SendCluster_Submit()
else:
	print 'Unknwon case'



