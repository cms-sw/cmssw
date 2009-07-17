#! /usr/bin/env python

import os
import sys
import fileinput
import string

##########################################################################
##########################################################################
######### User variables

#Reference release

RefRelease='CMSSW_3_1_1'

#Relval release (set if different from $CMSSW_VERSION)
NewRelease='CMSSW_3_1_1'

# startup and ideal sample list

#This are the standard relvals (startup)
#startupsamples= ['RelValTTbar', 'RelValMinBias', 'RelValQCD_Pt_3000_3500']

#This is pileup sample
#startupsamples= ['RelValTTbar_Tauola']

startupsamples= []

#This are the standard relvals (ideal)
idealsamples= ['RelValSingleMuPt1', 'RelValSingleMuPt10', 'RelValSingleMuPt100', 'RelValSinglePiPt1', 'RelValSinglePiPt10', 'RelValSinglePiPt100', 'RelValSingleElectronPt35', 'RelValTTbar', 'RelValQCD_Pt_3000_3500','RelValMinBias']

#This is pileup sample
#idealsamples= ['RelValZmumuJets_Pt_20_300_GEN']


idealsamples= ['RelValSingleMuPt100']


#
SeedCollections=['newSeedFromPairs', 'newSeedFromTriplets', 'secTriplets', 'thPLSeeds', 'fourthPLSeeds', 'fifthSeeds']

# Sequence. Possible values:
#   -only_validation
#   -re_tracking
#   -digi2track
#   -only_validation_and_TP
#   -re_tracking_and_TP
#   -digi2track_and_TP
#   -harvesting

Sequence='re_tracking'
#Sequence='harvesting'

# Ideal and Statup tags
IdealTag='MC_31X_V2'
StartupTag='STARTUP31X_V1'

# PileUp: PU . No PileUp: noPU
PileUp='noPU'

# Reference directory name (the macro will search for ReferenceSelection_Quality_Algo)
ReferenceSelection='IDEAL_31X_'+PileUp
StartupReferenceSelection='STARTUP_31X_'+PileUp

# Default label is GlobalTag_noPU__Quality_Algo. Change this variable if you want to append an additional string.
NewSelectionLabel=''


#Reference and new repository
RefRepository = '/afs/cern.ch/cms/performance/tracker/activities/reconstruction/tracking_performance/seeds'
NewRepository = '.'
#NewRepository = '/afs/cern.ch/cms/performance/tracker/activities/reconstruction/tracking_performance/seeds'

#Default Nevents
defaultNevents ='1000'

#Put here the number of event to be processed for specific samples (numbers must be strings) if not specified is -1:
Events={}

# template file names. Usually should not be changed.
cfg='seedPerformanceValidation_cfg.py'
macro='macro/SeedValHistoPublisher.C'



#########################################################################
#########################################################################
############ Functions

def replace(map, filein, fileout):
    replace_items = map.items()
    while 1:
        line = filein.readline()
        if not line: break
        for old, new in replace_items:
            line = string.replace(line, old, new)
        fileout.write(line)
    fileout.close()
    filein.close()
    
############################################

    
def do_validation(samples, GlobalTag):
    global Sequence, RefSelection, RefRepository, NewSelection, NewRepository, defaultNevents, Events
    global cfg, macro, Tracksname
    print 'Tag: ' + GlobalTag

    mineff='0.5'
    maxeff='1.025'
    maxfake='0.7'


    #build the New Selection name
    NewSelection=GlobalTag + '_' + PileUp

    NewSelection+=NewSelectionLabel

    #loop on all the requested samples
    for sample in samples :
        templatecfgFile = open(cfg, 'r')

        print 'Get information from DBS for sample', sample
        newdir=NewRepository+'/'+NewRelease+'/'+NewSelection+'/'+sample
	cfgFileName=sample
        #check if the sample is already done
        missing=False
        for seedcollection in SeedCollections :
            if(os.path.isfile(newdir+'/'+seedcollection+'/building.pdf' )!=True):
                  missing=True
        if(missing==True):
            #if the job is harvesting check if the file is already harvested
            harvestedfile='./DQM_V0001_R000000001__' + GlobalTag+ '__' + sample + '__Validation.root'
            if(( Sequence=="harvesting" and os.path.isfile(harvestedfile) )==False):
                    #search the primary dataset
                    cmd='dbsql "find  dataset.createdate, dataset where dataset like *'
            #            cmd+=sample+'/'+NewRelease+'_'+GlobalTag+'*GEN-SIM-DIGI-RAW-HLTDEBUG-RECO* "'
                    cmd+=sample+'/'+NewRelease+'_'+GlobalTag+'*GEN-SIM-RECO* "'
                    cmd+='|grep '+sample+'|grep -v test|sort|tail -1|cut -f2 '
                    print cmd
                    dataset= os.popen(cmd).readline().strip()
                    print 'DataSet:  ', dataset, '\n'

                    #Check if a dataset is found
                    if dataset!="":

                            #Find and format the list of files
                            cmd2='dbsql "find file where dataset like '+ dataset +'"|grep ' + sample
                            filenames='import FWCore.ParameterSet.Config as cms\n'
                            filenames+='readFiles = cms.untracked.vstring()\n'
                            filenames+='secFiles = cms.untracked.vstring()\n'
                            filenames+='source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)\n'
                            filenames+='readFiles.extend( [\n'
                            first=True
                            print cmd2
                            for line in os.popen(cmd2).readlines():
                                filename=line.strip()
                                if first==True:
                                    filenames+="'"
                                    filenames+=filename
                                    filenames+="'"
                                    first=False
                                else :
                                    filenames+=",\n'"
                                    filenames+=filename
                                    filenames+="'"
                            filenames+='])\n'

                            # if not harvesting find secondary file names
                            if(Sequence!="harvesting"):
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
                                            filenames+=']);\n'
                                    else :
                                            print "No primary dataset found skipping sample: ", sample
                                            continue
                            else :
                                    filenames+='secFiles.extend( (               ) )\n'

                            cfgFile = open(cfgFileName+'.py' , 'w' )
                            cfgFile.write(filenames)

                            if (Events.has_key(sample)!=True):
                                    Nevents=defaultNevents
                            else:
                                    Nevents=Events[sample]
                            firstseed=True
                            for seedcollection in SeedCollections :
                                if(firstseed):
                                    Seeds='cms.InputTag("'+seedcollection+'")'
                                    firstseed=False
                                else:
                                   Seeds+=',cms.InputTag("'+seedcollection+'")' 
                            symbol_map = { 'NEVENT':Nevents, 'GLOBALTAG':GlobalTag, 'SEQUENCE':Sequence, 'SAMPLE': sample, 'SEEDS':Seeds}


                            cfgFile = open(cfgFileName+'.py' , 'a' )
                            replace(symbol_map, templatecfgFile, cfgFile)

                            cmdrun='cmsRun ' +cfgFileName+ '.py >&  ' + cfgFileName + '.log < /dev/zero '
                            #cmdrun='date'
                            retcode=os.system(cmdrun)

                    else:      
                            print 'No dataset found skipping sample: '+ sample, '\n'  
                            continue
            else: 
                    retcode=0
            if (retcode!=0):
                    print 'Job for sample '+ sample + ' failed. \n'
            else:
                    for seedcollection in SeedCollections :
                        rootcommand='root -b -q -l CopySubdir.C\\('+ '\\\"val.' +sample+'.root\\\",\\\"val.' +sample+'_'+seedcollection+'.root\\\",\\\"'+ seedcollection+'_AssociatorByHits\\\",\\\"Seed\\\"\\) >& /dev/null'
                        print rootcommand
                        os.system(rootcommand)
                        referenceSample=RefRepository+'/'+RefRelease+'/'+RefSelection+'/'+sample+'/'+seedcollection + '/' + 'val.'+sample+'.root'
                        if os.path.isfile(referenceSample ):
                                replace_map = { 'NEW_FILE':'val.'+sample+'_'+seedcollection+'.root', 'REF_FILE':RefRelease+'/'+RefSelection+'/val.'+sample+'_'+seedcollection+'.root', 'REF_LABEL':sample, 'NEW_LABEL': sample, 'REF_RELEASE':RefRelease, 'NEW_RELEASE':NewRelease, 'REFSELECTION':RefSelection, 'NEWSELECTION':NewSelection, 'TrackValHistoPublisher': sample+'_'+seedcollection, 'MINEFF':mineff, 'MAXEFF':maxeff, 'MAXFAKE':maxfake}

                                if(os.path.exists(RefRelease+'/'+RefSelection)==False):
                                        os.makedirs(RefRelease+'/'+RefSelection)
                                os.system('cp ' + referenceSample+ ' '+RefRelease+'/'+RefSelection)  
                        else:
                                print "No reference file found at: ", RefRelease+'/'+RefSelection
                                replace_map = { 'NEW_FILE':'val.'+sample+'_'+seedcollection+'.root', 'REF_FILE':'val.'+sample+'_'+seedcollection+'.root', 'REF_LABEL':sample, 'NEW_LABEL': sample, 'REF_RELEASE':NewRelease, 'NEW_RELEASE':NewRelease, 'REFSELECTION':NewSelection, 'NEWSELECTION':NewSelection, 'TrackValHistoPublisher': sample+'_'+seedcollection, 'MINEFF':mineff, 'MAXEFF':maxeff, 'MAXFAKE':maxfake}

                        templatemacroFile = open(macro, 'r')
                        macroFile = open(cfgFileName+'_'+seedcollection+'.C' , 'w' )
                        replace(replace_map, templatemacroFile, macroFile)


                        os.system('root -b -q -l '+ cfgFileName+'_'+seedcollection+'.C'+ '>  macro.'+cfgFileName+'.log')
                        dir=newdir+'/'+seedcollection

                        if(os.path.exists(dir)==False):
                                os.makedirs(dir)

                        print "moving pdf files for sample: " , sample
                        os.system('mv  *.pdf ' + dir)

                        print "copying root file for sample: " , sample
                        os.system('cp val.'+ sample+ '.root ' + dir)

                        print "copy py file for sample: " , sample
                        os.system('cp '+cfgFileName+'.py ' + dir)
	
	
        else:
            print 'Validation for sample ' + sample + ' already done. Skipping this sample. \n'




##########################################################################
#########################################################################
######## This is the main program
if(NewRelease==''): 
    try:
        #Get some environment variables to use
        NewRelease     = os.environ["CMSSW_VERSION"]
        
    except KeyError:
        print >>sys.stderr, 'Error: The environment variable CMSSW_VERSION is not available.'
        print >>sys.stderr, '       Please run cmsenv'
        sys.exit()
else:
    try:
        #Get some environment variables to use
        os.environ["CMSSW_VERSION"]
        
    except KeyError:
        print >>sys.stderr, 'Error: CMSSW environment variables are not available.'
        print >>sys.stderr, '       Please run cmsenv'
        sys.exit()



NewSelection=''

RefSelection=ReferenceSelection
do_validation(idealsamples, IdealTag)
RefSelection=StartupReferenceSelection
do_validation(startupsamples, StartupTag)
        
