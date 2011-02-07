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
		Arguments[3] = WP_Pt
	 	WP_I=WP_Pt+0.5
		while WP_I>=WP_Pt-0.5:
			if WP_I>0:
				WP_I=WP_I-0.5
				continue
			Arguments[4] = WP_I
			if Arguments[9]==2:
				WP_TOF=WP_Pt+0.5
				while WP_TOF>=WP_Pt-0.5:
					if WP_TOF>0:
						WP_TOF=WP_TOF-0.5
						continue
					Arguments[5] = WP_TOF
					LaunchOnCondor.SendCluster_Push(Arguments)
					#print Arguments
					WP_TOF=WP_TOF-0.5
			else:
				Arguments[5] = 0.0
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
        LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"CUTFINDER"', 0.0, 0.0, 0.0, 0,'"dedxASmi"','"dedxCNPHarm2"',0]) #TkOnly
        LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"CUTFINDER"', 0.0, 0.0, 0.0, 0,'"dedxASmi"','"dedxCNPHarm2"',1]) #TkMuon
	LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"CUTFINDER"', 0.0, 0.0, 0.0, 0,'"dedxASmi"','"dedxCNPHarm2"',2]) #GLobalMuon
	LaunchOnCondor.SendCluster_Submit()

if sys.argv[1]=='1':
        print 'ANALYSIS'
        FarmDirectory = "FARM"
        JobName = "HscpAnalysis"
        LaunchOnCondor.Jobs_RunHere = 1
        LaunchOnCondor.SendCluster_Create(FarmDirectory, JobName)
        #NO CUTS!
        LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"',  0.0,  0.0,  0.0, 0,'"dedxASmi"','"dedxCNPHarm2"',0]) #TkOnly     No    Splitting
        LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"',  0.0,  0.0,  0.0, 1,'"dedxASmi"','"dedxCNPHarm2"',0]) #TkOnly     Hit   Splitting
        LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"',  0.0,  0.0,  0.0, 2,'"dedxASmi"','"dedxCNPHarm2"',0]) #TkOnly     HitEtaSplitting
        LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"',  0.0,  0.0,  0.0, 0,'"dedxASmi"','"dedxCNPHarm2"',1]) #TkMuon     No    Splitting
        LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"',  0.0,  0.0,  0.0, 1,'"dedxASmi"','"dedxCNPHarm2"',1]) #TkMuon     Hit   Splitting
        LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"',  0.0,  0.0,  0.0, 2,'"dedxASmi"','"dedxCNPHarm2"',1]) #TkMuon     HitEtaSplitting
        LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"',  0.0,  0.0,  0.0, 0,'"dedxASmi"','"dedxCNPHarm2"',2]) #GLobalMuon

        LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"', -0.5,  0.0,  0.0, 0,'"dedxASmi"','"dedxCNPHarm2"',2]) #GLobalMuon
        LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"',  0.0, -0.5,  0.0, 0,'"dedxASmi"','"dedxCNPHarm2"',2]) #GLobalMuon
        LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"', -0.5, -0.5,  0.0, 0,'"dedxASmi"','"dedxCNPHarm2"',2]) #GLobalMuon
        LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"',  0.0,  0.0, -0.5, 0,'"dedxASmi"','"dedxCNPHarm2"',2]) #GLobalMuon
        LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"', -0.5,  0.0, -0.5, 0,'"dedxASmi"','"dedxCNPHarm2"',2]) #GLobalMuon
        LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"',  0.0, -0.5, -0.5, 0,'"dedxASmi"','"dedxCNPHarm2"',2]) #GLobalMuon
        LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"', -0.5, -0.5, -0.5, 0,'"dedxASmi"','"dedxCNPHarm2"',2]) #GLobalMuon

        #VERY LOOSE CUTS 10^-2 --> Keep ~10^4 tracks!
        LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"', -1.0, -1.0, -0.0, 0,'"dedxASmi"','"dedxCNPHarm2"',0]) #TkOnly     No    Splitting
        LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"', -1.0, -1.0, -0.0, 1,'"dedxASmi"','"dedxCNPHarm2"',0]) #TkOnly     Hit   Splitting
        LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"', -1.0, -1.0, -0.0, 2,'"dedxASmi"','"dedxCNPHarm2"',0]) #TkOnly     HitEtaSplitting
        LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"', -1.0, -1.0, -0.0, 0,'"dedxASmi"','"dedxCNPHarm2"',1]) #TkMuon     No    Splitting
        LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"', -1.0, -1.0, -0.0, 1,'"dedxASmi"','"dedxCNPHarm2"',1]) #TkMuon     Hit   Splitting
        LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"', -1.0, -1.0, -0.0, 2,'"dedxASmi"','"dedxCNPHarm2"',1]) #TkMuon     HitEtaSplitting
        LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"', -0.5, -1.0, -0.5, 0,'"dedxASmi"','"dedxCNPHarm2"',2]) #GLobalMuon

        #VERY LOOSE CUTS 10^-3 --> Keep ~10^3 tracks!
        LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"', -1.5, -1.5, -0.0, 0,'"dedxASmi"','"dedxCNPHarm2"',0]) #TkOnly     No    Splitting
        LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"', -1.5, -1.5, -0.0, 1,'"dedxASmi"','"dedxCNPHarm2"',0]) #TkOnly     Hit   Splitting
        LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"', -1.5, -1.5, -0.0, 2,'"dedxASmi"','"dedxCNPHarm2"',0]) #TkOnly     HitEtaSplitting
        LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"', -1.5, -1.5, -0.0, 0,'"dedxASmi"','"dedxCNPHarm2"',1]) #TkMuon     No    Splitting
        LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"', -1.5, -1.5, -0.0, 1,'"dedxASmi"','"dedxCNPHarm2"',1]) #TkMuon     Hit   Splitting
        LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"', -1.5, -1.5, -0.0, 2,'"dedxASmi"','"dedxCNPHarm2"',1]) #TkMuon     HitEtaSplitting
        LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"', -1.0, -1.0, -1.0, 0,'"dedxASmi"','"dedxCNPHarm2"',2]) #GLobalMuon


        #LOOSE CUTS 10^-4 --> Keep ~10^2 tracks!
        LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"', -2.0, -2.0, -0.0, 0,'"dedxASmi"','"dedxCNPHarm2"',0]) #TkOnly     No    Splitting
        LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"', -2.0, -2.0, -0.0, 1,'"dedxASmi"','"dedxCNPHarm2"',0]) #TkOnly     Hit   Splitting
        LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"', -2.0, -2.0, -0.0, 2,'"dedxASmi"','"dedxCNPHarm2"',0]) #TkOnly     HitEtaSplitting
        LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"', -2.0, -2.0, -0.0, 0,'"dedxASmi"','"dedxCNPHarm2"',1]) #TkMuon     No    Splitting
        LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"', -2.0, -2.0, -0.0, 1,'"dedxASmi"','"dedxCNPHarm2"',1]) #TkMuon     Hit   Splitting
        LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"', -2.0, -2.0, -0.0, 2,'"dedxASmi"','"dedxCNPHarm2"',1]) #TkMuon     HitEtaSplitting
        LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"', -1.0, -2.0, -1.0, 0,'"dedxASmi"','"dedxCNPHarm2"',2]) #GLobalMuon


        #LOOSE CUTS 10^-5--> Keep ~10^1 tracks!
        LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"', -2.5, -2.5, -0.0, 0,'"dedxASmi"','"dedxCNPHarm2"',0]) #TkOnly     No    Splitting
        LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"', -2.5, -2.5, -0.0, 1,'"dedxASmi"','"dedxCNPHarm2"',0]) #TkOnly     Hit   Splitting
        LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"', -2.5, -2.5, -0.0, 2,'"dedxASmi"','"dedxCNPHarm2"',0]) #TkOnly     HitEtaSplitting
        LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"', -2.5, -2.5, -0.0, 0,'"dedxASmi"','"dedxCNPHarm2"',1]) #TkMuon     No    Splitting
        LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"', -2.5, -2.5, -0.0, 1,'"dedxASmi"','"dedxCNPHarm2"',1]) #TkMuon     Hit   Splitting
        LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"', -2.5, -2.5, -0.0, 2,'"dedxASmi"','"dedxCNPHarm2"',1]) #TkMuon     HitEtaSplitting
        LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"', -1.5, -2.0, -1.5, 0,'"dedxASmi"','"dedxCNPHarm2"',2]) #GLobalMuon


        #TIGHT CUTS 10^-6 --> Keep ~10^0 tracks!
        LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"', -3.0, -3.0, -0.0, 0,'"dedxASmi"','"dedxCNPHarm2"',0]) #TkOnly     No    Splitting
        LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"', -3.0, -3.0, -0.0, 1,'"dedxASmi"','"dedxCNPHarm2"',0]) #TkOnly     Hit   Splitting
        LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"', -3.0, -3.0, -0.0, 2,'"dedxASmi"','"dedxCNPHarm2"',0]) #TkOnly     HitEtaSplitting
        LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"', -3.0, -3.0, -0.0, 0,'"dedxASmi"','"dedxCNPHarm2"',1]) #TkMuon     No    Splitting
        LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"', -3.0, -3.0, -0.0, 1,'"dedxASmi"','"dedxCNPHarm2"',1]) #TkMuon     Hit   Splitting
        LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"', -3.0, -3.0, -0.0, 2,'"dedxASmi"','"dedxCNPHarm2"',1]) #TkMuon     HitEtaSplitting
        LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"', -2.0, -2.0, -2.0, 0,'"dedxASmi"','"dedxCNPHarm2"',2]) #GLobalMuon

        #TIGHT CUTS 10^-7 --> Keep ~10^-1 tracks!
        LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"', -3.5, -3.5, -0.0, 0,'"dedxASmi"','"dedxCNPHarm2"',0]) #TkOnly     No    Splitting
        LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"', -3.5, -3.5, -0.0, 1,'"dedxASmi"','"dedxCNPHarm2"',0]) #TkOnly     Hit   Splitting
        LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"', -3.5, -3.5, -0.0, 2,'"dedxASmi"','"dedxCNPHarm2"',0]) #TkOnly     HitEtaSplitting
        LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"', -3.5, -3.5, -0.0, 0,'"dedxASmi"','"dedxCNPHarm2"',1]) #TkMuon     No    Splitting
        LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"', -3.5, -3.5, -0.0, 1,'"dedxASmi"','"dedxCNPHarm2"',1]) #TkMuon     Hit   Splitting
        LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"', -3.5, -3.5, -0.0, 2,'"dedxASmi"','"dedxCNPHarm2"',1]) #TkMuon     HitEtaSplitting
        LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"', -2.0, -3.0, -2.0, 0,'"dedxASmi"','"dedxCNPHarm2"',2]) #GLobalMuon

        #TIGHT CUTS 10^-8 --> Keep ~10^-2 tracks!
        LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"', -4.0, -4.0, -0.0, 0,'"dedxASmi"','"dedxCNPHarm2"',0]) #TkOnly     No    Splitting
        LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"', -4.0, -4.0, -0.0, 1,'"dedxASmi"','"dedxCNPHarm2"',0]) #TkOnly     Hit   Splitting
        LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"', -4.0, -4.0, -0.0, 2,'"dedxASmi"','"dedxCNPHarm2"',0]) #TkOnly     HitEtaSplitting
        LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"', -4.0, -4.0, -0.0, 0,'"dedxASmi"','"dedxCNPHarm2"',1]) #TkMuon     No    Splitting
        LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"', -4.0, -4.0, -0.0, 1,'"dedxASmi"','"dedxCNPHarm2"',1]) #TkMuon     Hit   Splitting
        LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"', -4.0, -4.0, -0.0, 2,'"dedxASmi"','"dedxCNPHarm2"',1]) #TkMuon     HitEtaSplitting
        LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"', -2.5, -3.0, -2.5, 0,'"dedxASmi"','"dedxCNPHarm2"',2]) #GLobalMuon


        LaunchOnCondor.SendCluster_Submit()




if sys.argv[1]=='2':
        print 'ANALYSIS'
        FarmDirectory = "FARM"
        JobName = "HscpAnalysis"
        LaunchOnCondor.Jobs_RunHere = 1
        LaunchOnCondor.SendCluster_Create(FarmDirectory, JobName)
        ScanWP(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"', "WP_Pt", "WP_I", "WP_TOF", 0,'"dedxASmi"','"dedxCNPHarm2"',0]) #TkOnly     No    Splitting
        ScanWP(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"', "WP_Pt", "WP_I", "WP_TOF", 0,'"dedxASmi"','"dedxCNPHarm2"',1]) #TkMuon     No    Splitting
        ScanWP(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"', "WP_Pt", "WP_I", "WP_TOF", 0,'"dedxASmi"','"dedxCNPHarm2"',2]) #TkTOF      No    Splitting
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



