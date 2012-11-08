#!/usr/bin/env python

import urllib
import string
import os
import sys
import LaunchOnCondor
import glob

CMSSW_VERSION = os.getenv('CMSSW_VERSION','CMSSW_VERSION')
if CMSSW_VERSION == 'CMSSW_VERSION':
  print 'please setup your CMSSW environement'
  sys.exit(0)

if len(sys.argv)==1:
   print "Please pass in argument a number between 0 and 2"
   print "  0 - Run TriggerStudy.C"
   print "  1 - Run TriggerEfficiency.C"
   sys.exit()

elif sys.argv[1]=='0':
   print 'STUDY'
   FarmDirectory = "FARM"
   JobName = "HSCPTRIGGERSTUDY"
   LaunchOnCondor.Jobs_RunHere = 1
   LaunchOnCondor.SendCluster_Create(FarmDirectory, JobName)
   LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/TriggerStudy.C", '"summary_8TeV_Gluino"', '"Gluino_8TeV_M400_f10"', '"Gluino_8TeV_M800_f10"', '"Gluino_8TeV_M1200_f10"'])
   LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/TriggerStudy.C", '"summary_8TeV_GluinoN"', '"Gluino_8TeV_M400N_f10"', '"Gluino_8TeV_M800N_f10"', '"Gluino_8TeV_M1200N_f10"'])
   LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/TriggerStudy.C", '"summary_8TeV_Gluino_f100"', '"Gluino_8TeV_M400_f100"', '"Gluino_8TeV_M800_f100"', '"Gluino_8TeV_M1200_f100"'])
   LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/TriggerStudy.C", '"summary_8TeV_Stop"', '"Stop_8TeV_M200"', '"Stop_8TeV_M500"', '"Stop_8TeV_M800"'])
   LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/TriggerStudy.C", '"summary_8TeV_StopN"', '"Stop_8TeV_M200N"', '"Stop_8TeV_M500N"', '"Stop_8TeV_M800N"'])
   LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/TriggerStudy.C", '"summary_8TeV_GMStau"', '"GMStau_8TeV_M100"', '"GMStau_8TeV_M308"', '"GMStau_8TeV_M494"'])
   LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/TriggerStudy.C", '"summary_8TeV_PPStau"', '"PPStau_8TeV_M100"', '"PPStau_8TeV_M200"', '"PPStau_8TeV_M494"'])
   LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/TriggerStudy.C", '"summary_8TeV_DYLQ"', '"DY_8TeV_M100_Q1o3"', '"DY_8TeV_M600_Q1o3"', '"DY_8TeV_M100_Q2o3"', '"DY_8TeV_M600_Q2o3"'])
   LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/TriggerStudy.C", '"summary_8TeV_DYHQ"', '"DY_8TeV_M100_Q2"', '"DY_8TeV_M600_Q2"', '"DY_8TeV_M100_Q5"', '"DY_8TeV_M600_Q5"'])

   LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/TriggerStudy.C", '"summary_7TeV_Gluino"', '"Gluino_7TeV_M400_f10"', '"Gluino_7TeV_M800_f10"', '"Gluino_7TeV_M1200_f10"'])
   LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/TriggerStudy.C", '"summary_7TeV_GluinoN"', '"Gluino_7TeV_M400N_f10"', '"Gluino_7TeV_M800N_f10"', '"Gluino_7TeV_M1200N_f10"'])
   LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/TriggerStudy.C", '"summary_7TeV_Gluino_f100"', '"Gluino_7TeV_M400_f100"', '"Gluino_7TeV_M800_f100"', '"Gluino_7TeV_M1200_f100"'])
   LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/TriggerStudy.C", '"summary_7TeV_Stop"', '"Stop_7TeV_M200"', '"Stop_7TeV_M500"', '"Stop_7TeV_M800"'])
   LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/TriggerStudy.C", '"summary_7TeV_StopN"', '"Stop_7TeV_M200N"', '"Stop_7TeV_M500N"', '"Stop_7TeV_M800N"'])
   LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/TriggerStudy.C", '"summary_7TeV_GMStau"', '"GMStau_7TeV_M100"', '"GMStau_7TeV_M308"', '"GMStau_7TeV_M494"'])
   LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/TriggerStudy.C", '"summary_7TeV_PPStau"', '"PPStau_7TeV_M100"', '"PPStau_7TeV_M200"', '"PPStau_7TeV_M494"'])
   LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/TriggerStudy.C", '"summary_7TeV_DYLQ"', '"DY_7TeV_M100_Q1o3"', '"DY_7TeV_M600_Q1o3"', '"DY_7TeV_M100_Q2o3"', '"DY_7TeV_M600_Q2o3"'])
   LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/TriggerStudy.C", '"summary_7TeV_DYHQ"', '"DY_7TeV_M100_Q2"', '"DY_7TeV_M600_Q2"', '"DY_7TeV_M100_Q5"', '"DY_7TeV_M600_Q5"'])
   LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/TriggerStudy.C", '"test"', '"DY_7TeV_M100_Q1"', '"DY_7TeV_M100_Q3"', '"DY_7TeV_M100_Q5"'])

   LaunchOnCondor.SendCluster_Submit()

elif sys.argv[1]=='1':
   print 'EFFICIENCIENCY'
   FarmDirectory = "FARM"
   JobName = "HSCPEFFICIENCYCheck"
   LaunchOnCondor.Jobs_RunHere = 1
   LaunchOnCondor.SendCluster_Create(FarmDirectory, JobName)
   f= open('../../ICHEP_Analysis/Analysis_Samples.txt','r')
   index = -1
   for line in f :
      index+=1
      vals=line.split(',')
      if((vals[0].replace('"','')) in CMSSW_VERSION and int(vals[1])==0):
         LaunchOnCondor.SendCluster_Push(["FWLITE", os.getcwd()+"/TriggerEfficiency.C", '"ANALYSE_'+str(index)+'_to_'+str(index)+'"'])
   f.close()
   LaunchOnCondor.SendCluster_Submit()
