#!/usr/bin/env python

import urllib
import string
import os
import sys
import LaunchOnCondor
import glob


def ScanWP(Arguments):
	WP_Pt=0.0
	while WP_Pt>=-4.0:
		Arguments[8] = WP_Pt
	 	WP_I=WP_Pt
		while WP_I>=WP_Pt-1.0:
			if WP_I>0:
				WP_I=WP_I-0.5
				continue
			Arguments[9] = WP_I
			if Arguments[3]==2:
				WP_TOF=WP_Pt
				while WP_TOF>=WP_Pt-1.0:
					if (WP_TOF>0) or (WP_Pt+WP_I+WP_TOF<-7.0):
						WP_TOF=WP_TOF-0.5
						continue
					Arguments[10] = WP_TOF
					LaunchOnCondor.SendCluster_Push(Arguments)
					#print Arguments
					WP_TOF=WP_TOF-0.5
			else:
				Arguments[10] = 0.0
				LaunchOnCondor.SendCluster_Push(Arguments)
				#print Arguments
			WP_I=WP_I-0.5
		WP_Pt=WP_Pt-0.5
			

if len(sys.argv)==1:
	print "Please pass in argument a number between 0 and 2"
	sys.exit()

if sys.argv[1]=='0':	
        print 'ANALYSIS'
        FarmDirectory = "FARM"
        JobName = "HscpCutFinder"
	LaunchOnCondor.Jobs_RunHere = 1
	LaunchOnCondor.SendCluster_Create(FarmDirectory, JobName)	
        LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"CUTFINDER"', 0, 0, '"dedxASmi"'  ,'"dedxHarm2"', '"combined"', 0.0, 0.0, 0.0,  15.0]) #TkOnly Discrim Pt>15
        LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"CUTFINDER"', 2, 0, '"dedxASmi"'  ,'"dedxHarm2"', '"combined"', 0.0, 0.0, 0.0,  15.0]) #TkTOF  Discrim Pt>15
        LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"CUTFINDER"', 0, 0, '"dedxHarm2"' ,'"dedxHarm2"', '"combined"', 0.0, 0.0, 0.0,  15.0]) #TkOnly Estim   Pt>15
        LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"CUTFINDER"', 2, 0, '"dedxHarm2"' ,'"dedxHarm2"', '"combined"', 0.0, 0.0, 0.0,  15.0]) #TkTOF  Estim   Pt>15
        LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"CUTFINDER"', 2, 0, '"dedxASmi"'  ,'"dedxHarm2"', '"dt"'      , 0.0, 0.0, 0.0,  15.0]) #TkTOF  Discrim Pt>15


        LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"CUTFINDER"', 0, 0, '"dedxASmi"'  ,'"dedxHarm2"', '"combined"', 0.0, 0.0, 0.0,  20.0]) #TkOnly Discrim Pt>20
        LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"CUTFINDER"', 2, 0, '"dedxASmi"'  ,'"dedxHarm2"', '"combined"', 0.0, 0.0, 0.0,  20.0]) #TkTOF  Discrim Pt>20
        LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"CUTFINDER"', 0, 0, '"dedxHarm2"' ,'"dedxHarm2"', '"combined"', 0.0, 0.0, 0.0,  20.0]) #TkOnly Estim   Pt>20
        LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"CUTFINDER"', 2, 0, '"dedxHarm2"' ,'"dedxHarm2"', '"combined"', 0.0, 0.0, 0.0,  20.0]) #TkTOF  Estim   Pt>20
	LaunchOnCondor.SendCluster_Submit()

if sys.argv[1]=='1':
        print 'ANALYSIS'
        FarmDirectory = "FARM"
        JobName = "HscpAnalysis"
        LaunchOnCondor.Jobs_RunHere = 1
        LaunchOnCondor.SendCluster_Create(FarmDirectory, JobName)
        ScanWP(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"', 0, 0, '"dedxASmi"'  ,'"dedxHarm2"', '"combined"', "WP_Pt", "WP_I", "WP_TOF",  15.0]) #TkOnly Discrim Pt>15   No    Splitting
        ScanWP(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"', 2, 0, '"dedxASmi"'  ,'"dedxHarm2"', '"combined"', "WP_Pt", "WP_I", "WP_TOF",  15.0]) #TkTOF  Discrim Pt>15   No    Splitting
        ScanWP(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"', 0, 0, '"dedxHarm2"' ,'"dedxHarm2"', '"combined"', "WP_Pt", "WP_I", "WP_TOF",  15.0]) #TkOnly Estim   Pt>15   No    Splitting
        ScanWP(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"', 2, 0, '"dedxHarm2"' ,'"dedxHarm2"', '"combined"', "WP_Pt", "WP_I", "WP_TOF",  15.0]) #TkTOF  Estim   Pt>15   No    Splitting
        ScanWP(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"', 2, 0, '"dedxASmi"'  ,'"dedxHarm2"', '"dt"'      , "WP_Pt", "WP_I", "WP_TOF",  15.0]) #TkTOF  Discrim Pt>15   No    Splitting

        ScanWP(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"', 0, 0, '"dedxASmi"'  ,'"dedxHarm2"', '"combined"', "WP_Pt", "WP_I", "WP_TOF",  20.0]) #TkOnly Discrim Pt>20   No    Splitting
        ScanWP(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"', 2, 0, '"dedxASmi"'  ,'"dedxHarm2"', '"combined"', "WP_Pt", "WP_I", "WP_TOF",  20.0]) #TkTOF  Discrim Pt>20   No    Splitting
        ScanWP(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"', 0, 0, '"dedxHarm2"' ,'"dedxHarm2"', '"combined"', "WP_Pt", "WP_I", "WP_TOF",  20.0]) #TkOnly Estim   Pt>20   No    Splitting
        ScanWP(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"', 2, 0, '"dedxHarm2"' ,'"dedxHarm2"', '"combined"', "WP_Pt", "WP_I", "WP_TOF",  20.0]) #TkTOF  Estim   Pt>20   No    Splitting
        LaunchOnCondor.SendCluster_Submit()


if sys.argv[1]=='99':
	os.system('mkdir Results')
	os.system('mkdir Results/dedxASmi')	
	os.system('mkdir Results/dedxASmi/dt')	
	os.system('mkdir Results/dedxASmi/dt/Eta25')	
	os.system('mkdir Results/dedxASmi/dt/Eta25/Type0')	
	os.system('mkdir Results/dedxASmi/dt/Eta25/Type1')	
	os.system('mkdir Results/dedxASmi/dt/Eta25/Type2')	
	os.system('cp Save/dedxASmi/dt/Eta25/Type0/CutHistos.root Results/dedxASmi/dt/Eta25/Type0/.')
	os.system('cp Save/dedxASmi/dt/Eta25/Type1/CutHistos.root Results/dedxASmi/dt/Eta25/Type1/.')	
	os.system('cp Save/dedxASmi/dt/Eta25/Type2/CutHistos.root Results/dedxASmi/dt/Eta25/Type2/.')	

else:
	print 'Unknwon case: use an other argument'



