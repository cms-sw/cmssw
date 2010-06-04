#!/usr/bin/env python

import urllib
import string
import os
import sys
import LaunchOnCondor
import glob


SIGNAL='"MGStop200"'
if len(sys.argv)==1:
	print "Please pass in argument a number between 0 and 2"
	sys.exit()

if sys.argv[1]=='0':
        print 'MAP CREATION'
        FarmDirectory = "FARM"
        JobName = "HscpMap"
        LaunchOnCondor.SendCluster_Create(FarmDirectory, JobName)	
	WP_Pt=0
	while WP_Pt>=-5:
        	WP_I=0
	        while WP_I>=-5:
		        LaunchOnCondor.SendCluster_Push(["FWLITE", "Analysis_Step2345.C", '"MAKE_MAP"', WP_Pt, WP_I, 0, 0, 0, 0])
                        LaunchOnCondor.SendCluster_Push(["FWLITE", "Analysis_Step2345.C", '"MAKE_MAP"', WP_Pt, WP_I, 1, 0, 0, 0])
                        LaunchOnCondor.SendCluster_Push(["FWLITE", "Analysis_Step2345.C", '"MAKE_MAP"', WP_Pt, WP_I, 2, 0, 0, 0])
                        LaunchOnCondor.SendCluster_Push(["FWLITE", "Analysis_Step2345.C", '"MAKE_MAP"', WP_Pt, WP_I, 0, 5, 0, 0])
                        LaunchOnCondor.SendCluster_Push(["FWLITE", "Analysis_Step2345.C", '"MAKE_MAP"', WP_Pt, WP_I, 1, 5, 0, 0])
                        LaunchOnCondor.SendCluster_Push(["FWLITE", "Analysis_Step2345.C", '"MAKE_MAP"', WP_Pt, WP_I, 2, 5, 0, 0])

                        LaunchOnCondor.SendCluster_Push(["FWLITE", "Analysis_Step2345.C", '"MAKE_MAP"', WP_Pt, WP_I, 0, 0, 0, 1])
                        LaunchOnCondor.SendCluster_Push(["FWLITE", "Analysis_Step2345.C", '"MAKE_MAP"', WP_Pt, WP_I, 1, 0, 0, 1])
                        LaunchOnCondor.SendCluster_Push(["FWLITE", "Analysis_Step2345.C", '"MAKE_MAP"', WP_Pt, WP_I, 2, 0, 0, 1])
                        LaunchOnCondor.SendCluster_Push(["FWLITE", "Analysis_Step2345.C", '"MAKE_MAP"', WP_Pt, WP_I, 0, 5, 0, 1])
                        LaunchOnCondor.SendCluster_Push(["FWLITE", "Analysis_Step2345.C", '"MAKE_MAP"', WP_Pt, WP_I, 1, 5, 0, 1])
                        LaunchOnCondor.SendCluster_Push(["FWLITE", "Analysis_Step2345.C", '"MAKE_MAP"', WP_Pt, WP_I, 2, 5, 0, 1])
                	WP_I=WP_I-0.5
	        WP_Pt=WP_Pt-0.5
        LaunchOnCondor.SendCluster_Submit()
elif sys.argv[1]=='1':
	print 'MAP MERGING'
        FarmDirectory = "FARM"
        JobName       = "HscpMerge"
        LaunchOnCondor.SendCluster_Create(FarmDirectory, JobName)
        LaunchOnCondor.SendCluster_Push  (["FWLITE", "Analysis_Step2345.C", '"MERGE_MAP"', -1, -1, 0, 0, 0, 0])
        LaunchOnCondor.SendCluster_Push  (["FWLITE", "Analysis_Step2345.C", '"MERGE_MAP"', -1, -1, 1, 0, 0, 0])
        LaunchOnCondor.SendCluster_Push  (["FWLITE", "Analysis_Step2345.C", '"MERGE_MAP"', -1, -1, 2, 0, 0, 0])
        LaunchOnCondor.SendCluster_Push  (["FWLITE", "Analysis_Step2345.C", '"MERGE_MAP"', -1, -1, 0, 5, 0, 0])
        LaunchOnCondor.SendCluster_Push  (["FWLITE", "Analysis_Step2345.C", '"MERGE_MAP"', -1, -1, 1, 5, 0, 0])
        LaunchOnCondor.SendCluster_Push  (["FWLITE", "Analysis_Step2345.C", '"MERGE_MAP"', -1, -1, 2, 5, 0, 0])

        LaunchOnCondor.SendCluster_Push  (["FWLITE", "Analysis_Step2345.C", '"MERGE_MAP"', -1, -1, 0, 0, 0, 1])
        LaunchOnCondor.SendCluster_Push  (["FWLITE", "Analysis_Step2345.C", '"MERGE_MAP"', -1, -1, 1, 0, 0, 1])
        LaunchOnCondor.SendCluster_Push  (["FWLITE", "Analysis_Step2345.C", '"MERGE_MAP"', -1, -1, 2, 0, 0, 1])
        LaunchOnCondor.SendCluster_Push  (["FWLITE", "Analysis_Step2345.C", '"MERGE_MAP"', -1, -1, 0, 5, 0, 1])
        LaunchOnCondor.SendCluster_Push  (["FWLITE", "Analysis_Step2345.C", '"MERGE_MAP"', -1, -1, 1, 5, 0, 1])
        LaunchOnCondor.SendCluster_Push  (["FWLITE", "Analysis_Step2345.C", '"MERGE_MAP"', -1, -1, 2, 5, 0, 1])

        LaunchOnCondor.SendCluster_Submit()

elif sys.argv[1]=='2':
	print 'ANALYSIS'
        FarmDirectory = "FARM"
        JobName       = "HscpAnalysis"
	LaunchOnCondor.SendCluster_Create(FarmDirectory, JobName)
        LaunchOnCondor.SendCluster_Push  (["FWLITE", "Analysis_Step2345.C", '"ANALYSE"', -1.5, -2.0, 0, 0, 0, 0])
        LaunchOnCondor.SendCluster_Push  (["FWLITE", "Analysis_Step2345.C", '"ANALYSE"', -1.5, -2.0, 1, 0, 0, 0])
        LaunchOnCondor.SendCluster_Push  (["FWLITE", "Analysis_Step2345.C", '"ANALYSE"', -1.5, -2.0, 2, 0, 0, 0])
        LaunchOnCondor.SendCluster_Push  (["FWLITE", "Analysis_Step2345.C", '"ANALYSE"', -1.5, -2.0, 0, 5, 0, 0])
        LaunchOnCondor.SendCluster_Push  (["FWLITE", "Analysis_Step2345.C", '"ANALYSE"', -1.5, -2.0, 1, 5, 0, 0])
        LaunchOnCondor.SendCluster_Push  (["FWLITE", "Analysis_Step2345.C", '"ANALYSE"', -1.5, -2.0, 2, 5, 0, 0])
        LaunchOnCondor.SendCluster_Push  (["FWLITE", "Analysis_Step2345.C", '"ANALYSE"', -2.0, -2.5, 0, 0, 0, 0])
        LaunchOnCondor.SendCluster_Push  (["FWLITE", "Analysis_Step2345.C", '"ANALYSE"', -2.0, -2.5, 1, 0, 0, 0])
        LaunchOnCondor.SendCluster_Push  (["FWLITE", "Analysis_Step2345.C", '"ANALYSE"', -2.0, -2.5, 2, 0, 0, 0])
        LaunchOnCondor.SendCluster_Push  (["FWLITE", "Analysis_Step2345.C", '"ANALYSE"', -2.0, -2.5, 0, 5, 0, 0])
        LaunchOnCondor.SendCluster_Push  (["FWLITE", "Analysis_Step2345.C", '"ANALYSE"', -2.0, -2.5, 1, 5, 0, 0])
        LaunchOnCondor.SendCluster_Push  (["FWLITE", "Analysis_Step2345.C", '"ANALYSE"', -2.0, -2.5, 2, 5, 0, 0])
        LaunchOnCondor.SendCluster_Push  (["FWLITE", "Analysis_Step2345.C", '"ANALYSE"', -2.5, -3.0, 0, 0, 0, 0])
        LaunchOnCondor.SendCluster_Push  (["FWLITE", "Analysis_Step2345.C", '"ANALYSE"', -2.5, -3.0, 1, 0, 0, 0])
        LaunchOnCondor.SendCluster_Push  (["FWLITE", "Analysis_Step2345.C", '"ANALYSE"', -2.5, -3.0, 2, 0, 0, 0])
        LaunchOnCondor.SendCluster_Push  (["FWLITE", "Analysis_Step2345.C", '"ANALYSE"', -2.5, -3.0, 0, 5, 0, 0])
        LaunchOnCondor.SendCluster_Push  (["FWLITE", "Analysis_Step2345.C", '"ANALYSE"', -2.5, -3.0, 1, 5, 0, 0])
        LaunchOnCondor.SendCluster_Push  (["FWLITE", "Analysis_Step2345.C", '"ANALYSE"', -2.5, -3.0, 2, 5, 0, 0])

        LaunchOnCondor.SendCluster_Push  (["FWLITE", "Analysis_Step2345.C", '"ANALYSE"', -1.5, -2.0, 0, 1, 1, 0])
        LaunchOnCondor.SendCluster_Push  (["FWLITE", "Analysis_Step2345.C", '"ANALYSE"', -1.5, -2.0, 1, 1, 1, 0])
        LaunchOnCondor.SendCluster_Push  (["FWLITE", "Analysis_Step2345.C", '"ANALYSE"', -1.5, -2.0, 2, 1, 1, 0])
        LaunchOnCondor.SendCluster_Push  (["FWLITE", "Analysis_Step2345.C", '"ANALYSE"', -1.5, -2.0, 0, 5, 1, 0])
        LaunchOnCondor.SendCluster_Push  (["FWLITE", "Analysis_Step2345.C", '"ANALYSE"', -1.5, -2.0, 1, 5, 1, 0])
        LaunchOnCondor.SendCluster_Push  (["FWLITE", "Analysis_Step2345.C", '"ANALYSE"', -1.5, -2.0, 2, 5, 1, 0])
        LaunchOnCondor.SendCluster_Push  (["FWLITE", "Analysis_Step2345.C", '"ANALYSE"', -2.0, -2.5, 0, 1, 1, 0])
        LaunchOnCondor.SendCluster_Push  (["FWLITE", "Analysis_Step2345.C", '"ANALYSE"', -2.0, -2.5, 1, 1, 1, 0])
        LaunchOnCondor.SendCluster_Push  (["FWLITE", "Analysis_Step2345.C", '"ANALYSE"', -2.0, -2.5, 2, 1, 1, 0])
        LaunchOnCondor.SendCluster_Push  (["FWLITE", "Analysis_Step2345.C", '"ANALYSE"', -2.0, -2.5, 0, 5, 1, 0])
        LaunchOnCondor.SendCluster_Push  (["FWLITE", "Analysis_Step2345.C", '"ANALYSE"', -2.0, -2.5, 1, 5, 1, 0])
        LaunchOnCondor.SendCluster_Push  (["FWLITE", "Analysis_Step2345.C", '"ANALYSE"', -2.0, -2.5, 2, 5, 1, 0])
        LaunchOnCondor.SendCluster_Push  (["FWLITE", "Analysis_Step2345.C", '"ANALYSE"', -2.5, -3.0, 0, 1, 1, 0])
        LaunchOnCondor.SendCluster_Push  (["FWLITE", "Analysis_Step2345.C", '"ANALYSE"', -2.5, -3.0, 1, 1, 1, 0])
        LaunchOnCondor.SendCluster_Push  (["FWLITE", "Analysis_Step2345.C", '"ANALYSE"', -2.5, -3.0, 2, 1, 1, 0])
        LaunchOnCondor.SendCluster_Push  (["FWLITE", "Analysis_Step2345.C", '"ANALYSE"', -2.5, -3.0, 0, 5, 1, 0])
        LaunchOnCondor.SendCluster_Push  (["FWLITE", "Analysis_Step2345.C", '"ANALYSE"', -2.5, -3.0, 1, 5, 1, 0])
        LaunchOnCondor.SendCluster_Push  (["FWLITE", "Analysis_Step2345.C", '"ANALYSE"', -2.5, -3.0, 2, 5, 1, 0])

        LaunchOnCondor.SendCluster_Push  (["FWLITE", "Analysis_Step2345.C", '"ANALYSE"', -1.5, -2.0, 0, 0, 0, 1])
        LaunchOnCondor.SendCluster_Push  (["FWLITE", "Analysis_Step2345.C", '"ANALYSE"', -1.5, -2.0, 1, 0, 0, 1])
        LaunchOnCondor.SendCluster_Push  (["FWLITE", "Analysis_Step2345.C", '"ANALYSE"', -1.5, -2.0, 2, 0, 0, 1])
        LaunchOnCondor.SendCluster_Push  (["FWLITE", "Analysis_Step2345.C", '"ANALYSE"', -1.5, -2.0, 0, 5, 0, 1])
        LaunchOnCondor.SendCluster_Push  (["FWLITE", "Analysis_Step2345.C", '"ANALYSE"', -1.5, -2.0, 1, 5, 0, 1])
        LaunchOnCondor.SendCluster_Push  (["FWLITE", "Analysis_Step2345.C", '"ANALYSE"', -1.5, -2.0, 2, 5, 0, 1])
        LaunchOnCondor.SendCluster_Push  (["FWLITE", "Analysis_Step2345.C", '"ANALYSE"', -2.0, -2.5, 0, 0, 0, 1])
        LaunchOnCondor.SendCluster_Push  (["FWLITE", "Analysis_Step2345.C", '"ANALYSE"', -2.0, -2.5, 1, 0, 0, 1])
        LaunchOnCondor.SendCluster_Push  (["FWLITE", "Analysis_Step2345.C", '"ANALYSE"', -2.0, -2.5, 2, 0, 0, 1])
        LaunchOnCondor.SendCluster_Push  (["FWLITE", "Analysis_Step2345.C", '"ANALYSE"', -2.0, -2.5, 0, 5, 0, 1])
        LaunchOnCondor.SendCluster_Push  (["FWLITE", "Analysis_Step2345.C", '"ANALYSE"', -2.0, -2.5, 1, 5, 0, 1])
        LaunchOnCondor.SendCluster_Push  (["FWLITE", "Analysis_Step2345.C", '"ANALYSE"', -2.0, -2.5, 2, 5, 0, 1])
        LaunchOnCondor.SendCluster_Push  (["FWLITE", "Analysis_Step2345.C", '"ANALYSE"', -2.5, -3.0, 0, 0, 0, 1])
        LaunchOnCondor.SendCluster_Push  (["FWLITE", "Analysis_Step2345.C", '"ANALYSE"', -2.5, -3.0, 1, 0, 0, 1])
        LaunchOnCondor.SendCluster_Push  (["FWLITE", "Analysis_Step2345.C", '"ANALYSE"', -2.5, -3.0, 2, 0, 0, 1])
        LaunchOnCondor.SendCluster_Push  (["FWLITE", "Analysis_Step2345.C", '"ANALYSE"', -2.5, -3.0, 0, 5, 0, 1])
        LaunchOnCondor.SendCluster_Push  (["FWLITE", "Analysis_Step2345.C", '"ANALYSE"', -2.5, -3.0, 1, 5, 0, 1])
        LaunchOnCondor.SendCluster_Push  (["FWLITE", "Analysis_Step2345.C", '"ANALYSE"', -2.5, -3.0, 2, 5, 0, 1])

        LaunchOnCondor.SendCluster_Submit()
else:
	print 'Unknwon case'



