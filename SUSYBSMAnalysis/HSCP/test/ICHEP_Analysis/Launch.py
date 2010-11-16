#!/usr/bin/env python

import urllib
import string
import os
import sys
import LaunchOnCondor
import glob


def ScanWP(Arguments):
	WP_Pt=0.0
	while WP_Pt>=-5.0:
	 	WP_I=0.0
		while WP_I>=-5.0:
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
	#This is used to move the result from the node machine to the storage machine
#	LaunchOnCondor.Jobs_FinalCmds = ["ls Results/*/*/*/*/*/*/*/* -lth", "cp -rdfv Results " + os.getcwd() + "/."]
	LaunchOnCondor.Jobs_RunHere = 1
	LaunchOnCondor.SendCluster_Create(FarmDirectory, JobName)	
#	ScanWP(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"', "WP_Pt", "WP_I", 2, 11, 3, 1])
#	ScanWP(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"', "WP_Pt", "WP_I", 2, 11, 3, 0])
#	LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"', 0.0, 0.0, 2, 8, 0, 0])
#	LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"', 0.0, 0.0, 2, 8, 0, 1])

	ScanWP(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"', "WP_Pt", "WP_I", 1, 11, 3, 1, 2.5, 0.15])
	ScanWP(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"', "WP_Pt", "WP_I", 1, 11, 3, 0, 2.5, 0.15])
	LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"', 0.0, 0.0, 1, 8, 0, 0, 2.5, 0.15])
	LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"', 0.0, 0.0, 1, 8, 0, 1, 2.5, 0.15])

#        ScanWP(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"', "WP_Pt", "WP_I", 1, 11, 3, 1, 1.0, 0.15])
#        ScanWP(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"', "WP_Pt", "WP_I", 1, 11, 3, 0, 1.0, 0.15])
#        LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"', 0.0, 0.0, 1, 8, 0, 0, 1.0, 0.15])
#        LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"', 0.0, 0.0, 1, 8, 0, 1, 1.0, 0.15])


	ScanWP(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"', "WP_Pt", "WP_I", 0, 11, 3, 1, 2.5, 0.15])
	ScanWP(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"', "WP_Pt", "WP_I", 0, 11, 3, 0, 2.5, 0.15])
	LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"', 0.0, 0.0, 0, 8, 0, 0, 2.5, 0.15])
	LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"', 0.0, 0.0, 0, 8, 0, 1, 2.5, 0.15])

#	ScanWP(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"', "WP_Pt", "WP_I", 1, 3, 3, 1])
#	ScanWP(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"', "WP_Pt", "WP_I", 1, 3, 3, 0])

#        ScanWP(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"', "WP_Pt", "WP_I", 2, 3, 3, 1])
#        ScanWP(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"', "WP_Pt", "WP_I", 2, 3, 3, 0])


	LaunchOnCondor.SendCluster_Submit()




if sys.argv[1]=='1':
        print 'ANALYSIS'
        FarmDirectory = "FARM"
        JobName = "HscpAnalysis"
        #This is used to move the result from the node machine to the storage machine
#       LaunchOnCondor.Jobs_FinalCmds = ["ls Results/*/*/*/*/*/*/*/* -lth", "cp -rdfv Results " + os.getcwd() + "/."]
        LaunchOnCondor.Jobs_RunHere = 1
        LaunchOnCondor.SendCluster_Create(FarmDirectory, JobName)
        ScanWP(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"', "WP_Pt", "WP_I", 1, 11, 3, 1])
        ScanWP(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"', "WP_Pt", "WP_I", 1, 11, 3, 0])
        LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"', 0.0, 0.0, 1, 8, 0, 0])
        LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"', 0.0, 0.0, 1, 8, 0, 1])
        LaunchOnCondor.SendCluster_Submit()


if sys.argv[1]=='2':
        print 'ANALYSIS'
        FarmDirectory = "FARM"
        JobName = "HscpAnalysis"
        #This is used to move the result from the node machine to the storage machine
#       LaunchOnCondor.Jobs_FinalCmds = ["ls Results/*/*/*/*/*/*/*/* -lth", "cp -rdfv Results " + os.getcwd() + "/."]
        LaunchOnCondor.Jobs_RunHere = 1
        LaunchOnCondor.SendCluster_Create(FarmDirectory, JobName)
        ScanWP(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"', "WP_Pt", "WP_I", 0, 11, 3, 1])
        ScanWP(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"', "WP_Pt", "WP_I", 0, 11, 3, 0])
        LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"', 0.0, 0.0, 0, 8, 0, 0])
        LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"', 0.0, 0.0, 0, 8, 0, 1])
        LaunchOnCondor.SendCluster_Submit()

else:
	print 'Unknwon case: use an other argument'



