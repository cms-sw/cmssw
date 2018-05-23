#! /usr/bin/env python

import os
import sys
import fileinput
import string

##########################################################################
##########################################################################
######### User variables

#Reference release
NewRelease='CMSSW_3_8_0_pre2'

# startup and ideal sample list
#startupsamples= ['RelValSingleMuPt10', 'RelValSingleMuPt100', 'RelValSingleMuPt1000', 'RelValTTbar','RelValZMM']
startupsamples= []

#idealsamples= ['RelValSingleMuPt10', 'RelValSingleMuPt100', 'RelValSingleMuPt1000', 'RelValTTbar','RelValZMM']
idealsamples= ['RelValSingleMuPt100']

# track algorithm name and quality. Can be a list.
Algos= ['']
Qualities=['']
#Qualities=['', 'highPurity']

#Leave unchanged unless the track collection name changed
Tracksname=''

# Sequence. Possible values:
#   -harvesting
#   -reco_and_val
#   -only_val

Sequence='reco_and_val'

SearchContent="*GEN-SIM-RECO*"
DriverSteps="HARVESTING:validationHarvesting"
if(Sequence=="harvesting"):
    SearchContent="*GEN-SIM-RECO*"
    DriverSteps="HARVESTING:validationHarvesting+dqmHarvesting"
if(Sequence=="reco_and_val"):
    SearchContent="*GEN-SIM*HLTDEBUG*"
    DriverSteps="RAW2DIGI,RECO,VALIDATION"
if(Sequence=="only_val"):
    SearchContent="*GEN-SIM-RECO*"
    DriverSteps="POSTRECO,VALIDATION"

print Sequence+' '+SearchContent

Submit=False
DBS=True
OneAtATime=False

# Ideal and Statup tags
IdealTag='MC_38Y_V1'
StartupTag='STARTUP_31X_v1'

# Default label is GlobalTag_noPU__Quality_Algo. Change this variable if you want to append an additional string.
NewSelectionLabel='_test'

WorkDirBase = '/tmp/'
#WorkDirBase = '/tmp/aperrott'
#WorkDirBase = '/afs/cern.ch/user/a/aeverett/scratch0'

#Default Nevents
defaultNevents ='-1'

#Put here the number of event to be processed for specific samples (numbers must be strings) if not specified is -1:
Events={} #{ 'RelValZMM':'5000', 'RelValTTbar':'5000'}



#########################################################################
#########################################################################
############ Functions

def replace(map, filein, fileout):
    replace_items = map.items()
    while True:
        line = filein.readline()
        if not line: break
        for old, new in replace_items:
            line = string.replace(line, old, new)
        fileout.write(line)
    fileout.close()
    filein.close()
    
############################################

    
def do_validation(samples, GlobalTagUse, trackquality, trackalgorithm):
    global Sequence, NewSelection, defaultNevents, Events
    global Tracksname,SearchContent,DriverSteps
    splitResult = GlobalTagUse.split('_')
    GlobalTag = splitResult[0]
    print 'Search Tag: ' + GlobalTag
    print 'Tag to use: ' + GlobalTagUse

    #build the New Selection name
    NewSelection=GlobalTag +'_noPU'
    if( trackquality !=''):
        NewSelection+='_'+trackquality
    if(trackalgorithm!=''):
        NewSelection+='_'+trackalgorithm
    if(trackquality =='') and (trackalgorithm==''):
        if(Tracksname==''):
            NewSelection+='_ootb'
            Tracks='generalTracks'
        else:
           NewSelection+= Tracks
    elif(Tracksname==''):
        Tracks='cutsRecoTracks'
    else:
        Tracks=Tracksname
    NewSelection+=NewSelectionLabel

    #loop on all the requested samples
    for sample in samples :
        print 'Get information from DBS for sample', sample

        WorkDir = WorkDirBase+'/'+NewRelease+'/'+NewSelection+'/'+sample

        if(os.path.exists(WorkDir)==False):
            os.makedirs(WorkDir)

        
        #chech if the sample is already done

        if(True):
            ## start new
            cmd='dbsql "find  dataset where dataset like *'
            cmd+=sample+'/'+NewRelease+'_'+GlobalTag+SearchContent+' "'
            cmd+='|grep '+sample+'|grep -v test|sort|tail -1|cut -f2 '
            print cmd
            dataset= os.popen(cmd).readline().strip()
            print 'DataSet:  ', dataset, '\n'
            #Check if a dataset is found
            if(dataset!="" or DBS==False):
		
                #Find and format the list of files
                cmd2='dbsql "find file where dataset like '+ dataset +'"|grep ' + sample
                thisFile=0
                filenames='import FWCore.ParameterSet.Config as cms\n'
                filenames+='readFiles = cms.untracked.vstring()\n'
                filenames+='secFiles = cms.untracked.vstring()\n'
                filenames+='source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)\n'
                filenames+='readFiles.extend( [\n'
                first=True
                print cmd2
                for line in os.popen(cmd2).readlines():
                    filename=line.strip()
                    thisFile=thisFile+1
                    if first==True:
                        filenames+="'"
                        filenames+=filename
                        filenames+="'"
                        first=False
                    else :
                        filenames+=",\n'"
                        filenames+=filename
                        filenames+="'"
                filenames+='\n]);\n'
				
                # if not harvesting find secondary file names
                if(Sequence!="harvesting" and Sequence!="reco_and_val"):
                        cmd3='dbsql  "find dataset.parent where dataset like '+ dataset +'"|grep ' + sample
                        parentdataset=os.popen(cmd3).readline()
                        print 'Parent DataSet:  ', parentdataset, '\n'
                    
                    #Check if a dataset is found
                        if parentdataset!="":
                            cmd4='dbsql  "find file where dataset like '+ parentdataset +'"|grep ' + sample 
                            filenames+='secFiles.extend( [\n'
                            first=True
		
                            for line in os.popen(cmd4).readlines():
                                secfilename=line.strip()
                                if first==True:
                                    filenames+="'"
                                    filenames+=secfilename
                                    filenames+="'"
                                    first=False
                                else :
                                    filenames+=",\n'"
                                    filenames+=secfilename
                                    filenames+="'"
                            filenames+='\n]);\n'
                        else :
                            print "No parent dataset found skipping sample: ", sample
                            filenames+='secFiles.extend( (               ) )\n'

                            #continue
                else :
                        filenames+='secFiles.extend( (               ) )\n'

            ## end new


                cfgFileName=('%s_%s_%d') % (sample,Sequence,thisFile)
                print 'cfgFileName ' + cfgFileName
                
                #sampleFileName=('sample_%s_%d') % (sample,thisFile)
                sampleFileName='sample_'+Sequence
                sampleFile = open(sampleFileName+'.py','w' )
                sampleFile.write(filenames)
                sampleFile.close()
                #print filenames
                
                if ((sample in Events)!=True):
                    Nevents=defaultNevents
                else:
                    Nevents=Events[sample]
                
                symbol_map = { 'NEVENT':Nevents, 'GLOBALTAG':GlobalTagUse, 'SEQUENCE':Sequence, 'SAMPLE': sample, 'ALGORITHM':trackalgorithm, 'QUALITY':trackquality, 'TRACKS':Tracks}

                cmdrun='cmsRun ' + WorkDir +'/'+cfgFileName+ '.py >&  ' + cfgFileName + '.log < /dev/zero '

                print cmdrun
                
                lancialines='#!/bin/bash \n'
                lancialines+='cd '+ProjectBase+'/src \n'
                lancialines+='eval `scramv1 run -sh` \n\n'
                lancialines+='export PYTHONPATH=.:$PYTHONPATH \n'
                lancialines+='cd '+WorkDir+'\n'
                lancialines+='cmsRun '+cfgFileName+'.py  >&  ' + cfgFileName + '.log < /dev/zero \n'
#                lancialines+='mv  DQM_V0001_R000000001__' + GlobalTagUse+ '__' + sample + '__Validation.root' + ' ' + 'val.' +sample+'.root \n'
                lancialines+='mv  DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root' + ' ' + 'val.' +sample+'.root \n'
                
                lanciaName=('lancia_%s_%s_%s_%d') % (GlobalTag,sample,Sequence,thisFile)
                lanciaFile = open(lanciaName,'w')
                lanciaFile.write(lancialines)
                lanciaFile.close()

                previousFileOut=''
                command=''
                nextFileOut=''
                specialCommand=''
                processName=''
                if(Sequence=="reco_and_val"):
                    nextFileOut+='step2_RECO_VALIDATION.root'
                    command+='step2'
                    previousFileOut+='step1.root'
                    processName+='RecVal'
                if(Sequence=="only_val"):
                    nextFileOut+='step2_VALIDATION.root'
                    command+='step2'
                    previousFileOut+='step2.root'
                    processName+='VAL'
                if(Sequence=="harvesting"):
                    if(previousFileOut==''):
                        previousFileOut+='step2.root'
                    command+='harvest'
                    nextFileOut+='harvest_out.root'
                    specialCommand+=' --harvesting AtJobEnd'
                    processName+='HARV'

                driverCmdNew='cmsDriver.py '+ command
                driverCmdNew+=' -s '+DriverSteps
                driverCmdNew+=' '+processName+' '
                driverCmdNew+=' --mc '
                driverCmdNew+=' -n'+Nevents
                driverCmdNew+=' --conditions FrontierConditions_GlobalTag,'+GlobalTagUse+'::All'
                driverCmdNew+=' --filein file:'+previousFileOut
                driverCmdNew+=' --fileout '+nextFileOut
                driverCmdNew+=' --python_filename='+cfgFileName+'.py '
                driverCmdNew+=' --no_exec '
                driverCmdNew+=' --customise=Validation/RecoMuon/customise.py '
                driverCmdNew+=' '+specialCommand+' ' 

                print driverCmdNew
                iii=os.system(driverCmdNew)
                    
                print ("copying py file for sample: %s %s") % (sample,lanciaName)
                iii = os.system('mv '+lanciaName+' '+WorkDir+'/.')
                print iii
                iii = os.system('mv '+sampleFileName+'.py '+WorkDir)
                iii = os.system('mv '+cfgFileName+'.py '+WorkDir)
                #iii = os.system('mv harvest_'+cfgFileName+'.py '+WorkDir)
                #if(OneAtATime==False):
                #    break
                
                retcode=0

                if(Submit):
                    retcode=os.system(cmdrun)
                    if os.path.isfile(sample+'.log'):
                        os.system('mv '+sample+'.log ' + WorkDir)
                else:
                    continue

                if (retcode!=0):
                    print 'Job for sample '+ sample + ' failed. \n'
                else:
                    if Sequence=="harvesting":
                        os.system('mv  DQM_V0001_R000000001__' + GlobalTag+ '__' + sample + '__Validation.root' + ' ' + WorkDir+'/val.' +sample+'.root')
                    if os.path.isfile('fullOutput.'+sample+'.root'):
                        os.system('mv fullOutput.'+sample+'.root ' + WorkDir)
                    if os.path.isfile('output.'+sample+'.root'):
                        os.system('mv output.'+sample+'.root ' + WorkDir)
                        
            else:
                print 'No dataset found skipping sample: '+ sample, '\n'
        else:
            print 'Validation for sample ' + sample + ' already done. Skipping this sample. \n'




##########################################################################
#########################################################################
######## This is the main program
            
try:
     #Get some environment variables to use
     if Submit:
         print 'Will Submit job'
         #NewRelease     = os.environ["CMSSW_VERSION"]
     ProjectBase    = os.environ["CMSSW_BASE"]
     #else:
         #NewRelease='CMSSW_2_2_3'


      
except KeyError:
     print >>sys.stderr, 'Error: The environment variable CMSSW_VERSION is not available.'
     print >>sys.stderr, '       Please run eval `scramv1 runtime -csh` to set your environment variables'
     sys.exit()


#print 'Running validation on the following samples: ', samples

if os.path.isfile( 'dbsCommandLine.py')!=True:
    print >>sys.stderr, "Failed to find the file ($DBSCMD_HOME/dbsCommandLine.py)"

NewSelection=''

for algo in Algos:
    print 'algo ' + algo
    for quality in Qualities:
        print 'quality ' + quality
        do_validation(idealsamples, IdealTag, quality , algo)
        do_validation(startupsamples, StartupTag, quality , algo)

