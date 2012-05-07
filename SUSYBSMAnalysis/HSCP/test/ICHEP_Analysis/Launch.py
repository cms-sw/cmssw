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
		Arguments[8] = WP_Pt
	 	WP_I=WP_Pt+1.5
		while WP_I>=WP_Pt-1.5:
			if WP_I>0:
				WP_I=WP_I-0.5
				continue
			Arguments[9] = WP_I
			if Arguments[3]==2:
				WP_TOF=WP_Pt+1.5
				while WP_TOF>=WP_Pt-1.5:
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


def ScanCutId(Arguments):
	WP_Pt=30.0
	while WP_Pt<=150.0:
		Arguments[8] = WP_Pt
	 	WP_I=0.05
		while WP_I<=0.25:
			Arguments[9] = WP_I
			if Arguments[3]==2:
				WP_TOF=1.0
				while WP_TOF<=1.3:
					Arguments[10] = WP_TOF
					LaunchOnCondor.SendCluster_Push(Arguments)
					WP_TOF=WP_TOF+0.05
			else:
				Arguments[10] = 0.0
				LaunchOnCondor.SendCluster_Push(Arguments)
			WP_I=WP_I+0.03
		WP_Pt=WP_Pt+20

def ScanCutIe(Arguments):
	WP_Pt=30.0
	while WP_Pt<=150.0:
		Arguments[8] = WP_Pt
	 	WP_I=2.5
		while WP_I<=4.5:
			Arguments[9] = WP_I
			if Arguments[3]==2:
				WP_TOF=1.0
				while WP_TOF<=1.3:
					Arguments[10] = WP_TOF
					LaunchOnCondor.SendCluster_Push(Arguments)
					WP_TOF=WP_TOF+0.05
			else:
				Arguments[10] = 0.0
				LaunchOnCondor.SendCluster_Push(Arguments)
			WP_I=WP_I+0.25
		WP_Pt=WP_Pt+20

			

def ComputeLimits(InputPattern):
        LaunchOnCondor.SendCluster_Push(["ROOT", os.getcwd()+"/Analysis_Step6.C", '"ANALYSE"', InputPattern, '"Gluino200_2C"' , '"Gluino200"', 0.0    / 0.3029 , 0.0    / 0.4955 , 1.0    / 0.2015])
        LaunchOnCondor.SendCluster_Push(["ROOT", os.getcwd()+"/Analysis_Step6.C", '"ANALYSE"', InputPattern, '"Gluino300_2C"' , '"Gluino300"', 0.0    / 0.3029 , 0.0    / 0.4955 , 1.0    / 0.2015])
        LaunchOnCondor.SendCluster_Push(["ROOT", os.getcwd()+"/Analysis_Step6.C", '"ANALYSE"', InputPattern, '"Gluino400_2C"' , '"Gluino400"', 0.0    / 0.3029 , 0.0    / 0.4955 , 1.0    / 0.2015])
        LaunchOnCondor.SendCluster_Push(["ROOT", os.getcwd()+"/Analysis_Step6.C", '"ANALYSE"', InputPattern, '"Gluino500_2C"' , '"Gluino500"', 0.0    / 0.3029 , 0.0    / 0.4955 , 1.0    / 0.2015])
        LaunchOnCondor.SendCluster_Push(["ROOT", os.getcwd()+"/Analysis_Step6.C", '"ANALYSE"', InputPattern, '"Gluino600_2C"' , '"Gluino600"', 0.0    / 0.3029 , 0.0    / 0.4955 , 1.0    / 0.2015])
        LaunchOnCondor.SendCluster_Push(["ROOT", os.getcwd()+"/Analysis_Step6.C", '"ANALYSE"', InputPattern, '"Gluino900_2C"' , '"Gluino900"', 0.0    / 0.3029 , 0.0    / 0.4955 , 1.0    / 0.2015])

        LaunchOnCondor.SendCluster_Push(["ROOT", os.getcwd()+"/Analysis_Step6.C", '"ANALYSE"', InputPattern, '"Gluino200_f0"' , '"Gluino200"', 0.2524 / 0.3029 , 0.4893 / 0.4955 , 0.2583 / 0.2015])
        LaunchOnCondor.SendCluster_Push(["ROOT", os.getcwd()+"/Analysis_Step6.C", '"ANALYSE"', InputPattern, '"Gluino300_f0"' , '"Gluino300"', 0.2524 / 0.3029 , 0.4893 / 0.4955 , 0.2583 / 0.2015])
        LaunchOnCondor.SendCluster_Push(["ROOT", os.getcwd()+"/Analysis_Step6.C", '"ANALYSE"', InputPattern, '"Gluino400_f0"' , '"Gluino400"', 0.2524 / 0.3029 , 0.4893 / 0.4955 , 0.2583 / 0.2015])
        LaunchOnCondor.SendCluster_Push(["ROOT", os.getcwd()+"/Analysis_Step6.C", '"ANALYSE"', InputPattern, '"Gluino500_f0"' , '"Gluino500"', 0.2524 / 0.3029 , 0.4893 / 0.4955 , 0.2583 / 0.2015])
        LaunchOnCondor.SendCluster_Push(["ROOT", os.getcwd()+"/Analysis_Step6.C", '"ANALYSE"', InputPattern, '"Gluino600_f0"' , '"Gluino600"', 0.2524 / 0.3029 , 0.4893 / 0.4955 , 0.2583 / 0.2015])
        LaunchOnCondor.SendCluster_Push(["ROOT", os.getcwd()+"/Analysis_Step6.C", '"ANALYSE"', InputPattern, '"Gluino900_f0"' , '"Gluino900"', 0.2524 / 0.3029 , 0.4893 / 0.4955 , 0.2583 / 0.2015])

        LaunchOnCondor.SendCluster_Push(["ROOT", os.getcwd()+"/Analysis_Step6.C", '"ANALYSE"', InputPattern, '"Gluino200_f1"' , '"Gluino200"', 0.3029 / 0.3029 , 0.4955 / 0.4955 , 0.2015 / 0.2015])
        LaunchOnCondor.SendCluster_Push(["ROOT", os.getcwd()+"/Analysis_Step6.C", '"ANALYSE"', InputPattern, '"Gluino300_f1"' , '"Gluino300"', 0.3029 / 0.3029 , 0.4955 / 0.4955 , 0.2015 / 0.2015])
        LaunchOnCondor.SendCluster_Push(["ROOT", os.getcwd()+"/Analysis_Step6.C", '"ANALYSE"', InputPattern, '"Gluino400_f1"' , '"Gluino400"', 0.3029 / 0.3029 , 0.4955 / 0.4955 , 0.2015 / 0.2015])
        LaunchOnCondor.SendCluster_Push(["ROOT", os.getcwd()+"/Analysis_Step6.C", '"ANALYSE"', InputPattern, '"Gluino500_f1"' , '"Gluino500"', 0.3029 / 0.3029 , 0.4955 / 0.4955 , 0.2015 / 0.2015])
        LaunchOnCondor.SendCluster_Push(["ROOT", os.getcwd()+"/Analysis_Step6.C", '"ANALYSE"', InputPattern, '"Gluino600_f1"' , '"Gluino600"', 0.3029 / 0.3029 , 0.4955 / 0.4955 , 0.2015 / 0.2015])
        LaunchOnCondor.SendCluster_Push(["ROOT", os.getcwd()+"/Analysis_Step6.C", '"ANALYSE"', InputPattern, '"Gluino900_f1"' , '"Gluino900"', 0.3029 / 0.3029 , 0.4955 / 0.4955 , 0.2015 / 0.2015])

        LaunchOnCondor.SendCluster_Push(["ROOT", os.getcwd()+"/Analysis_Step6.C", '"ANALYSE"', InputPattern, '"Gluino200_f5"' , '"Gluino200"', 0.5739 / 0.3029 , 0.3704 / 0.4955 , 0.0557 / 0.2015])
        LaunchOnCondor.SendCluster_Push(["ROOT", os.getcwd()+"/Analysis_Step6.C", '"ANALYSE"', InputPattern, '"Gluino300_f5"' , '"Gluino300"', 0.5739 / 0.3029 , 0.3704 / 0.4955 , 0.0557 / 0.2015])
        LaunchOnCondor.SendCluster_Push(["ROOT", os.getcwd()+"/Analysis_Step6.C", '"ANALYSE"', InputPattern, '"Gluino400_f5"' , '"Gluino400"', 0.5739 / 0.3029 , 0.3704 / 0.4955 , 0.0557 / 0.2015])
        LaunchOnCondor.SendCluster_Push(["ROOT", os.getcwd()+"/Analysis_Step6.C", '"ANALYSE"', InputPattern, '"Gluino500_f5"' , '"Gluino500"', 0.5739 / 0.3029 , 0.3704 / 0.4955 , 0.0557 / 0.2015])
        LaunchOnCondor.SendCluster_Push(["ROOT", os.getcwd()+"/Analysis_Step6.C", '"ANALYSE"', InputPattern, '"Gluino600_f5"' , '"Gluino600"', 0.5739 / 0.3029 , 0.3704 / 0.4955 , 0.0557 / 0.2015])
        LaunchOnCondor.SendCluster_Push(["ROOT", os.getcwd()+"/Analysis_Step6.C", '"ANALYSE"', InputPattern, '"Gluino900_f5"' , '"Gluino900"', 0.5739 / 0.3029 , 0.3704 / 0.4955 , 0.0557 / 0.2015])

        LaunchOnCondor.SendCluster_Push(["ROOT", os.getcwd()+"/Analysis_Step6.C", '"ANALYSE"', InputPattern, '"Gluino200N_f0"', '"Gluino200N"', 0.2524 / 0.3029 , 0.4893 / 0.4955 , 0.2583 / 0.2015])
        LaunchOnCondor.SendCluster_Push(["ROOT", os.getcwd()+"/Analysis_Step6.C", '"ANALYSE"', InputPattern, '"Gluino300N_f0"', '"Gluino300N"', 0.2524 / 0.3029 , 0.4893 / 0.4955 , 0.2583 / 0.2015])
        LaunchOnCondor.SendCluster_Push(["ROOT", os.getcwd()+"/Analysis_Step6.C", '"ANALYSE"', InputPattern, '"Gluino400N_f0"', '"Gluino400N"', 0.2524 / 0.3029 , 0.4893 / 0.4955 , 0.2583 / 0.2015])
        LaunchOnCondor.SendCluster_Push(["ROOT", os.getcwd()+"/Analysis_Step6.C", '"ANALYSE"', InputPattern, '"Gluino500N_f0"', '"Gluino500N"', 0.2524 / 0.3029 , 0.4893 / 0.4955 , 0.2583 / 0.2015])
        LaunchOnCondor.SendCluster_Push(["ROOT", os.getcwd()+"/Analysis_Step6.C", '"ANALYSE"', InputPattern, '"Gluino600N_f0"', '"Gluino600N"', 0.2524 / 0.3029 , 0.4893 / 0.4955 , 0.2583 / 0.2015])
        LaunchOnCondor.SendCluster_Push(["ROOT", os.getcwd()+"/Analysis_Step6.C", '"ANALYSE"', InputPattern, '"Gluino900N_f0"', '"Gluino900N"', 0.2524 / 0.3029 , 0.4893 / 0.4955 , 0.2583 / 0.2015])

        LaunchOnCondor.SendCluster_Push(["ROOT", os.getcwd()+"/Analysis_Step6.C", '"ANALYSE"', InputPattern, '"Gluino200N_f1"', '"Gluino200N"', 0.3029 / 0.3029 , 0.4955 / 0.4955 , 0.2015 / 0.2015])
        LaunchOnCondor.SendCluster_Push(["ROOT", os.getcwd()+"/Analysis_Step6.C", '"ANALYSE"', InputPattern, '"Gluino300N_f1"', '"Gluino300N"', 0.3029 / 0.3029 , 0.4955 / 0.4955 , 0.2015 / 0.2015])
        LaunchOnCondor.SendCluster_Push(["ROOT", os.getcwd()+"/Analysis_Step6.C", '"ANALYSE"', InputPattern, '"Gluino400N_f1"', '"Gluino400N"', 0.3029 / 0.3029 , 0.4955 / 0.4955 , 0.2015 / 0.2015])
        LaunchOnCondor.SendCluster_Push(["ROOT", os.getcwd()+"/Analysis_Step6.C", '"ANALYSE"', InputPattern, '"Gluino500N_f1"', '"Gluino500N"', 0.3029 / 0.3029 , 0.4955 / 0.4955 , 0.2015 / 0.2015])
        LaunchOnCondor.SendCluster_Push(["ROOT", os.getcwd()+"/Analysis_Step6.C", '"ANALYSE"', InputPattern, '"Gluino600N_f1"', '"Gluino600N"', 0.3029 / 0.3029 , 0.4955 / 0.4955 , 0.2015 / 0.2015])
        LaunchOnCondor.SendCluster_Push(["ROOT", os.getcwd()+"/Analysis_Step6.C", '"ANALYSE"', InputPattern, '"Gluino900N_f1"', '"Gluino900N"', 0.3029 / 0.3029 , 0.4955 / 0.4955 , 0.2015 / 0.2015])

        LaunchOnCondor.SendCluster_Push(["ROOT", os.getcwd()+"/Analysis_Step6.C", '"ANALYSE"', InputPattern, '"Gluino200N_f5"', '"Gluino200N"', 0.5739 / 0.3029 , 0.3704 / 0.4955 , 0.0557 / 0.2015])
        LaunchOnCondor.SendCluster_Push(["ROOT", os.getcwd()+"/Analysis_Step6.C", '"ANALYSE"', InputPattern, '"Gluino300N_f5"', '"Gluino300N"', 0.5739 / 0.3029 , 0.3704 / 0.4955 , 0.0557 / 0.2015])
        LaunchOnCondor.SendCluster_Push(["ROOT", os.getcwd()+"/Analysis_Step6.C", '"ANALYSE"', InputPattern, '"Gluino400N_f5"', '"Gluino400N"', 0.5739 / 0.3029 , 0.3704 / 0.4955 , 0.0557 / 0.2015])
        LaunchOnCondor.SendCluster_Push(["ROOT", os.getcwd()+"/Analysis_Step6.C", '"ANALYSE"', InputPattern, '"Gluino500N_f5"', '"Gluino500N"', 0.5739 / 0.3029 , 0.3704 / 0.4955 , 0.0557 / 0.2015])
        LaunchOnCondor.SendCluster_Push(["ROOT", os.getcwd()+"/Analysis_Step6.C", '"ANALYSE"', InputPattern, '"Gluino600N_f5"', '"Gluino600N"', 0.5739 / 0.3029 , 0.3704 / 0.4955 , 0.0557 / 0.2015])
        LaunchOnCondor.SendCluster_Push(["ROOT", os.getcwd()+"/Analysis_Step6.C", '"ANALYSE"', InputPattern, '"Gluino900N_f5"', '"Gluino900N"', 0.5739 / 0.3029 , 0.3704 / 0.4955 , 0.0557 / 0.2015])

        LaunchOnCondor.SendCluster_Push(["ROOT", os.getcwd()+"/Analysis_Step6.C", '"ANALYSE"', InputPattern, '"Stop130_2C"'   , '"Stop130"', 0.0    / 0.1705 , 0.0    / 0.4868 , 1.0    / 0.3427])
        LaunchOnCondor.SendCluster_Push(["ROOT", os.getcwd()+"/Analysis_Step6.C", '"ANALYSE"', InputPattern, '"Stop200_2C"'   , '"Stop200"', 0.0    / 0.1705 , 0.0    / 0.4868 , 1.0    / 0.3427])
        LaunchOnCondor.SendCluster_Push(["ROOT", os.getcwd()+"/Analysis_Step6.C", '"ANALYSE"', InputPattern, '"Stop300_2C"'   , '"Stop300"', 0.0    / 0.1705 , 0.0    / 0.4868 , 1.0    / 0.3427])
        LaunchOnCondor.SendCluster_Push(["ROOT", os.getcwd()+"/Analysis_Step6.C", '"ANALYSE"', InputPattern, '"Stop500_2C"'   , '"Stop500"', 0.0    / 0.1705 , 0.0    / 0.4868 , 1.0    / 0.3427])
        LaunchOnCondor.SendCluster_Push(["ROOT", os.getcwd()+"/Analysis_Step6.C", '"ANALYSE"', InputPattern, '"Stop800_2C"'   , '"Stop800"', 0.0    / 0.1705 , 0.0    / 0.4868 , 1.0    / 0.3427])

        LaunchOnCondor.SendCluster_Push(["ROOT", os.getcwd()+"/Analysis_Step6.C", '"ANALYSE"', InputPattern, '"Stop130"'      , '"Stop130"', 0.1705 / 0.1705 , 0.4868 / 0.4868 , 0.3427 / 0.3427])
        LaunchOnCondor.SendCluster_Push(["ROOT", os.getcwd()+"/Analysis_Step6.C", '"ANALYSE"', InputPattern, '"Stop200"'      , '"Stop200"', 0.1705 / 0.1705 , 0.4868 / 0.4868 , 0.3427 / 0.3427])
        LaunchOnCondor.SendCluster_Push(["ROOT", os.getcwd()+"/Analysis_Step6.C", '"ANALYSE"', InputPattern, '"Stop300"'      , '"Stop300"', 0.1705 / 0.1705 , 0.4868 / 0.4868 , 0.3427 / 0.3427])
        LaunchOnCondor.SendCluster_Push(["ROOT", os.getcwd()+"/Analysis_Step6.C", '"ANALYSE"', InputPattern, '"Stop500"'      , '"Stop500"', 0.1705 / 0.1705 , 0.4868 / 0.4868 , 0.3427 / 0.3427])
        LaunchOnCondor.SendCluster_Push(["ROOT", os.getcwd()+"/Analysis_Step6.C", '"ANALYSE"', InputPattern, '"Stop800"'      , '"Stop800"', 0.1705 / 0.1705 , 0.4868 / 0.4868 , 0.3427 / 0.3427])

        LaunchOnCondor.SendCluster_Push(["ROOT", os.getcwd()+"/Analysis_Step6.C", '"ANALYSE"', InputPattern, '"Stop130N"'     , '"Stop130N"', 0.1705 / 0.1705 , 0.4868 / 0.4868 , 0.3427 / 0.3427])
        LaunchOnCondor.SendCluster_Push(["ROOT", os.getcwd()+"/Analysis_Step6.C", '"ANALYSE"', InputPattern, '"Stop200N"'     , '"Stop200N"', 0.1705 / 0.1705 , 0.4868 / 0.4868 , 0.3427 / 0.3427])
        LaunchOnCondor.SendCluster_Push(["ROOT", os.getcwd()+"/Analysis_Step6.C", '"ANALYSE"', InputPattern, '"Stop300N"'     , '"Stop300N"', 0.1705 / 0.1705 , 0.4868 / 0.4868 , 0.3427 / 0.3427])
        LaunchOnCondor.SendCluster_Push(["ROOT", os.getcwd()+"/Analysis_Step6.C", '"ANALYSE"', InputPattern, '"Stop500N"'     , '"Stop500N"', 0.1705 / 0.1705 , 0.4868 / 0.4868 , 0.3427 / 0.3427])
        LaunchOnCondor.SendCluster_Push(["ROOT", os.getcwd()+"/Analysis_Step6.C", '"ANALYSE"', InputPattern, '"Stop800N"'     , '"Stop800N"', 0.1705 / 0.1705 , 0.4868 / 0.4868 , 0.3427 / 0.3427])

        LaunchOnCondor.SendCluster_Push(["ROOT", os.getcwd()+"/Analysis_Step6.C", '"ANALYSE"', InputPattern, '"GMStau100"', '"GMStau100"'])
        LaunchOnCondor.SendCluster_Push(["ROOT", os.getcwd()+"/Analysis_Step6.C", '"ANALYSE"', InputPattern, '"GMStau126"', '"GMStau126"'])
        LaunchOnCondor.SendCluster_Push(["ROOT", os.getcwd()+"/Analysis_Step6.C", '"ANALYSE"', InputPattern, '"GMStau156"', '"GMStau156"'])
        LaunchOnCondor.SendCluster_Push(["ROOT", os.getcwd()+"/Analysis_Step6.C", '"ANALYSE"', InputPattern, '"GMStau200"', '"GMStau200"'])
        LaunchOnCondor.SendCluster_Push(["ROOT", os.getcwd()+"/Analysis_Step6.C", '"ANALYSE"', InputPattern, '"GMStau247"', '"GMStau247"'])
        LaunchOnCondor.SendCluster_Push(["ROOT", os.getcwd()+"/Analysis_Step6.C", '"ANALYSE"', InputPattern, '"GMStau308"', '"GMStau308"'])

        LaunchOnCondor.SendCluster_Push(["ROOT", os.getcwd()+"/Analysis_Step6.C", '"ANALYSE"', InputPattern, '"PPStau100"', '"PPStau100"'])
        LaunchOnCondor.SendCluster_Push(["ROOT", os.getcwd()+"/Analysis_Step6.C", '"ANALYSE"', InputPattern, '"PPStau126"', '"PPStau126"'])
        LaunchOnCondor.SendCluster_Push(["ROOT", os.getcwd()+"/Analysis_Step6.C", '"ANALYSE"', InputPattern, '"PPStau156"', '"PPStau156"'])
        LaunchOnCondor.SendCluster_Push(["ROOT", os.getcwd()+"/Analysis_Step6.C", '"ANALYSE"', InputPattern, '"PPStau200"', '"PPStau200"'])
        LaunchOnCondor.SendCluster_Push(["ROOT", os.getcwd()+"/Analysis_Step6.C", '"ANALYSE"', InputPattern, '"PPStau247"', '"PPStau247"'])
        LaunchOnCondor.SendCluster_Push(["ROOT", os.getcwd()+"/Analysis_Step6.C", '"ANALYSE"', InputPattern, '"PPStau308"', '"PPStau308"'])

        LaunchOnCondor.SendCluster_Push(["ROOT", os.getcwd()+"/Analysis_Step6.C", '"ANALYSE"', InputPattern, '"DCStau121"', '"DCStau121"'])
        LaunchOnCondor.SendCluster_Push(["ROOT", os.getcwd()+"/Analysis_Step6.C", '"ANALYSE"', InputPattern, '"DCStau182"', '"DCStau182"'])
        LaunchOnCondor.SendCluster_Push(["ROOT", os.getcwd()+"/Analysis_Step6.C", '"ANALYSE"', InputPattern, '"DCStau242"', '"DCStau242"'])
        LaunchOnCondor.SendCluster_Push(["ROOT", os.getcwd()+"/Analysis_Step6.C", '"ANALYSE"', InputPattern, '"DCStau302"', '"DCStau302"'])



if len(sys.argv)==1:
	print "Please pass in argument a number between 0 and 2"
	sys.exit()

if sys.argv[1]=='0':	
        print 'ANALYSIS'
        FarmDirectory = "FARM"
        JobName = "HscpAnalysis"
	LaunchOnCondor.Jobs_RunHere = 1
	LaunchOnCondor.SendCluster_Create(FarmDirectory, JobName)	
        LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"', 0, '"dedxASmi"'  ,'"dedxHarm2"'  , '"combined"', 0.0, 0.0, 0.0]) #TkOnly Discrim Pt>20
        LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"', 2, '"dedxASmi"'  ,'"dedxHarm2"'  , '"combined"', 0.0, 0.0, 0.0]) #TkTOF  Discrim Pt>20
#        LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"', 0, '"dedxASmi"'  ,'"dedxNPHarm2"', '"combined"', 0.0, 0.0, 0.0]) #TkOnly Discrim Pt>20
#        LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"', 2, '"dedxASmi"'  ,'"dedxNPHarm2"', '"combined"', 0.0, 0.0, 0.0]) #TkTOF  Discrim Pt>20
	LaunchOnCondor.SendCluster_Submit()

if sys.argv[1]=='1':
        print 'ANALYSIS'
        FarmDirectory = "FARM"
        JobName = "HscpAnalysis"
        LaunchOnCondor.Jobs_RunHere = 1
        LaunchOnCondor.SendCluster_Create(FarmDirectory, JobName)
        ScanWP(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"', 0, 0, '"dedxASmi"'  ,'"dedxHarm2"', '"combined"', "WP_Pt", "WP_I", "WP_TOF",  20.0]) #TkOnly Discrim Pt>20   No    Splitting
        ScanWP(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"', 2, 0, '"dedxASmi"'  ,'"dedxHarm2"', '"combined"', "WP_Pt", "WP_I", "WP_TOF",  20.0]) #TkTOF  Discrim Pt>20   No    Splitting
        LaunchOnCondor.SendCluster_Submit()



if sys.argv[1]=='2':
        print 'LIMITS'
        FarmDirectory = "FARM"
        JobName = "HscpLimits"
        LaunchOnCondor.Jobs_RunHere = 1
        LaunchOnCondor.SendCluster_Create(FarmDirectory, JobName)
        ComputeLimits('"Results/dedxASmi/combined/Eta25/PtMin20/Type0/"')
        ComputeLimits('"Results/dedxASmi/combined/Eta25/PtMin20/Type2/"')
        LaunchOnCondor.SendCluster_Submit()


if sys.argv[1]=='3':
        print 'ANALYSIS'
        FarmDirectory = "FARM"
        JobName = "HscpAnalysisCut"
        LaunchOnCondor.Jobs_RunHere = 1
        LaunchOnCondor.SendCluster_Create(FarmDirectory, JobName)
        ScanCutId(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"', 0, 0, '"dedxASmi"'  ,'"dedxHarm2"', '"combined"', "WP_Pt", "WP_I", "WP_TOF",  20.0]) #TkOnly Discrim Pt>20   No    Splitting
        ScanCutId(["FWLITE", os.getcwd()+"/Analysis_Step234.C", '"ANALYSE"', 2, 0, '"dedxASmi"'  ,'"dedxHarm2"', '"combined"', "WP_Pt", "WP_I", "WP_TOF",  20.0]) #TkTOF  Discrim Pt>20   No    Splitting
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



