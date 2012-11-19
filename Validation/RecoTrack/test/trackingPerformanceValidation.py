#! /usr/bin/env python

import os
import sys
import fileinput
import string
import copy

#########################################################
########### User Defined Variables (BEGIN) ##############


### Reference release
RefRelease='CMSSW_6_1_0_pre4'

### Relval release (set if different from $CMSSW_VERSION)
NewRelease='CMSSW_6_1_0_pre5'

### sample list 

### This is the list of IDEAL-conditions relvals 
startupsamples= [
'RelValMinBias',   ### list of samples to be validated for each pre-release  
'RelValQCD_Pt_3000_3500',
'RelValQCD_Pt_600_800',
'RelValSingleElectronPt35', 
'RelValSingleElectronPt10', 
'RelValTTbar', 
'RelValSingleMuPt10', 
'RelValSingleMuPt100',
]

pileupstartupsamples = [
#'RelValTTbar'
]

fastsimstartupsamples = [
'RelValTTbar'
]

pileupfastsimstartupsamples = [
#'RelValTTbar'
]

### Sample version: v1,v2,etc..
Version='v1'

# Global tags
StartupTag='START61_V4'

RefStartupTag='START61_V1'

### Track algorithm name and quality. Can be a list.
Algos= ['ootb', 'iter0', 'iter1','iter2','iter3','iter4','iter5','iter6']
Qualities=['', 'highPurity']

### Leave unchanged unless the track collection name changes
Tracksname=''

# Sequence. Possible values:
#   -only_validation
#   -re_tracking
#   -digi2track
#   -only_validation_and_TP
#   -re_tracking_and_TP
#   -digi2track_and_TP
#   -harvesting
#   -preproduction
#   -comparison_only


Sequence='comparison_only'



### Default label is GlobalTag_noPU__Quality_Algo. Change this variable if you want to append an additional string.
NewSelectionLabel=''


### Reference and new repository
RefRepository = '/afs/cern.ch/cms/Physics/tracking/validation/MC'
NewRepository = 'new' # copy output into a local folder


### Default Nevents
defaultNevents ='-1'

### Put here the number of event to be processed for specific samples (numbers must be strings) 
### if not specified is defaultNevents:
Events={}
#Events={'RelValTTbar':'4000'}

### template file names. Usually should not be changed.
cfg='trackingPerformanceValidation_cfg.py'
macro='macro/TrackValHistoPublisher.C'

########### User Defined Variables (END) ################
#########################################################






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

    
def do_validation(samples, GlobalTag, trackquality, trackalgorithm, PileUp, sampleType, dofastfull):
    global Sequence, Version, RefSelection, RefRepository, NewSelection, NewRepository, defaultNevents, Events, castorHarvestedFilesDirectory
    global cfg, macro, Tracksname
    tracks_map = { 'ootb':'general_AssociatorByHitsRecoDenom','iter0':'cutsRecoZero_AssociatorByHitsRecoDenom','iter1':'cutsRecoFirst_AssociatorByHitsRecoDenom','iter2':'cutsRecoSecond_AssociatorByHitsRecoDenom','iter3':'cutsRecoThird_AssociatorByHitsRecoDenom','iter4':'cutsRecoFourth_AssociatorByHitsRecoDenom','iter5':'cutsRecoFifth_AssociatorByHitsRecoDenom','iter6':'cutsRecoSixth_AssociatorByHitsRecoDenom'}
    tracks_map_hp = { 'ootb':'cutsRecoHp_AssociatorByHitsRecoDenom','iter0':'cutsRecoZeroHp_AssociatorByHitsRecoDenom','iter1':'cutsRecoFirstHp_AssociatorByHitsRecoDenom','iter2':'cutsRecoSecondHp_AssociatorByHitsRecoDenom','iter3':'cutsRecoThirdHp_AssociatorByHitsRecoDenom','iter4':'cutsRecoFourthHp_AssociatorByHitsRecoDenom','iter5':'cutsRecoFifthHp_AssociatorByHitsRecoDenom','iter6':'cutsRecoSixthHp_AssociatorByHitsRecoDenom'}
    if(trackalgorithm=='iter0' or trackalgorithm=='ootb'):
        mineff='0.0'
        maxeff='1.025'
        maxfake='0.7'
    elif(trackalgorithm=='iter1'):
        mineff='0.0'
        maxeff='0.5'
        maxfake='0.8'
    elif(trackalgorithm=='iter2'):
        mineff='0.0'
        maxeff='0.25'
        maxfake='0.8'
    elif(trackalgorithm=='iter4'):
        mineff='0.0'
        maxeff='0.3'
        maxfake='0.8'
    elif(trackalgorithm=='iter5' or trackalgorithm=='iter6'):
        mineff='0.0'
        maxeff='1.0'
        maxfake='0.8'
    else:
        mineff='0'
        maxeff='0.1'
        maxfake='0.8'
    #build the New Selection name
    NewSelection=GlobalTag + '_' + PileUp
    if( trackquality !=''):
        NewSelection+='_'+trackquality
    if(trackalgorithm!=''and not(trackalgorithm=='ootb' and trackquality !='')):
        NewSelection+='_'+trackalgorithm
    if(trackquality =='') and (trackalgorithm==''):
        if(Tracksname==''):
            NewSelection+='_ootb'
            Tracks='generalTracks'
        else:
           NewSelection+= Tracks
    if(Tracksname==''):
        Tracks='cutsRecoTracks'
    else:
        Tracks=Tracksname
    NewSelection+=NewSelectionLabel
    listofdatasets = open('listofdataset.txt' , 'w' )
    #loop on all the requested samples
    for sample in samples :
        templatecfgFile = open(cfg, 'r')
        templatemacroFile = open(macro, 'r')
        newdir=NewRepository+'/'+NewRelease+'/'+NewSelection+'/'+sample 
	cfgFileName=sample+GlobalTag
        #check if the sample is already done
        if(os.path.isfile(newdir+'/building.pdf' )!=True):    

            if( Sequence=="harvesting"):
            	harvestedfile='./DQM_V0001_R000000001__' + GlobalTag+ '__' + sample + '__Validation.root'
                print harvestedfile
            elif( Sequence=="preproduction"):
                harvestedfile='./DQM_V0001_R000000001__' + sample+ '-' + GlobalTag + '_preproduction_312-v1__GEN-SIM-RECO_1.root'
            elif( Sequence=="comparison_only"):
                if (sampleType == 'FullSim' and PileUp == 'noPU') : harvestedfile='./DQM_V0001_R000000001__' + sample+ '__' + NewRelease+ '-' +GlobalTag + '-' + Version + '__DQM.root'
                if (sampleType == 'FullSim' and PileUp == 'PU') : harvestedfile='./DQM_V0001_R000000001__' + sample+ '__' + NewRelease+ '-PU_' +GlobalTag + '-' + Version + '__DQM.root'
                if (sampleType == 'FastSim' and PileUp == 'noPU') : harvestedfile = './DQM_V0001_R000000001__' + sample+ '__' + NewRelease+ '-' +GlobalTag + '_FastSim-' + Version + '__GEN-SIM-DIGI-RECO.root'
                if (sampleType == 'FastSim' and PileUp == 'PU') : 
                    harvestedfile = './DQM_V0001_R000000001__' + sample+ '__' + NewRelease+ '-PU_' +GlobalTag + '_FastSim-' + Version + '__GEN-SIM-DIGI-RECO.root'

            print 'Sample:  ', sample, sampleType, PileUp, trackquality, trackalgorithm, '\n'

            if (Sequence != "comparison_only"):
                print 'Get information from DBS for sample', sample
                #search the primary dataset
                cmd='dbsql "find  dataset where dataset like /'
                if (sampleType == 'FullSim' and PileUp == 'noPU'): cmd+=sample+'/'+NewRelease+'-'+GlobalTag+'*'+Version+'/GEN-SIM-RECO order by dataset.createdate "'
                if (sampleType == 'FullSim' and PileUp == 'PU'):   cmd+=sample+'/'+NewRelease+'-PU_'+GlobalTag+'*'+Version+'/GEN-SIM-RECO order by dataset.createdate "'
                if (sampleType == 'FastSim' and PileUp == 'noPU'): cmd+=sample+'/'+NewRelease+'-'+GlobalTag+'_FastSim-'+Version+'/GEN-SIM-DIGI-RECO order by dataset.createdate "'
                if (sampleType == 'FastSim' and PileUp == 'PU'): cmd+=sample+'/'+NewRelease+'-PU_'+GlobalTag+'_FastSim-'+Version+'/GEN-SIM-DIGI-RECO order by dataset.createdate "'
                cmd+='|grep '+sample+'|grep -v test|tail -1'
                print cmd
                #Check if a dataset is found
                dataset= os.popen(cmd).readline().strip()
                print 'DataSet:  ', dataset, '\n'

                if dataset!="":
                        listofdatasets.write(dataset)
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
                        filenames+=']);\n'

                        # if not harvesting find secondary file names (only for FullSim samples)
                        if(Sequence!="preproduction" and sampleType=="FullSim"):
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
                                        filenames+='\n ]);\n'
                                else :
                                        print "No primary dataset found skipping sample: ", sample
                                        continue
                        else :
                                filenames+='secFiles.extend( (               ) )'

                        cfgFile = open(cfgFileName+'.py' , 'w' )
                        cfgFile.write(filenames)

                        if (Events.has_key(sample)!=True):
                                Nevents=defaultNevents
                        else:
                                Nevents=Events[sample]
                        thealgo=trackalgorithm
                        thequality=trackquality
                        if(trackalgorithm=='ootb'):
                            thealgo=''
                        if(thealgo!=''):
                            thealgo='\''+thealgo+'\''
                        if(trackquality!=''):
                            thequality='\''+trackquality+'\''
                        symbol_map = { 'NEVENT':Nevents, 'GLOBALTAG':GlobalTag, 'SEQUENCE':Sequence, 'SAMPLE': sample, 'ALGORITHM':thealgo, 'QUALITY':thequality, 'TRACKS':Tracks}


                        cfgFile = open(cfgFileName+'.py' , 'a' )
                        replace(symbol_map, templatecfgFile, cfgFile)
                        if(( (Sequence=="harvesting" or Sequence=="preproduction" or Sequence=="comparison_only") and os.path.isfile(harvestedfile) )==False):
                            # if the file is already harvested do not run the job again
                            #cmdrun='cmsRun ' +cfgFileName+ '.py >&  ' + cfgFileName + '.log < /dev/zero '
                            cmdrun='cmsRun ' +cfgFileName+ '.py'
                            retcode=os.system(cmdrun)
                        else:
                            retcode=0

                else:      
                        print 'No dataset found skipping sample: '+ sample, '\n'  
                        continue

                if (retcode!=0):
                       print 'Job for sample '+ sample + ' failed. \n'
            if (Sequence=="harvesting" or Sequence=="preproduction" or Sequence=="comparison_only"):
                    #copy only the needed histograms
                    if(trackquality==""):
                            rootcommand='root -b -q -l CopySubdir.C\\('+ '\\\"'+harvestedfile+'\\\",\\\"val.' +sample+'.root\\\",\\\"'+ tracks_map[trackalgorithm]+ '\\\"\\) >& /dev/null'
                            #rootcommand='root -b -q -l CopySubdir.C\\('+ '\\\"'+harvestedfile+'\\\",\\\"val.' +sample+'.root\\\",\\\"'+ tracks_map[trackalgorithm]+ '\\\"\\)'
                            os.system(rootcommand)
                    elif(trackquality=="highPurity"):
                            os.system('root -b -q -l CopySubdir.C\\('+ '\\\"'+harvestedfile+'\\\",\\\"val.' +sample+'.root\\\",\\\"'+ tracks_map_hp[trackalgorithm]+ '\\\"\\) >& /dev/null')
                            #os.system('root -b -q -l CopySubdir.C\\('+ '\\\"'+harvestedfile+'\\\",\\\"val.' +sample+'.root\\\",\\\"'+ tracks_map_hp[trackalgorithm]+ '\\\"\\)')

            if (sampleType == 'FastSim' and dofastfull == False): referenceSample=RefRepository+'/'+RefRelease+'/fastsim/'+RefRelease+'/'+RefSelection+'/'+sample+'/'+'val.'+sample+'.root'
            if (sampleType == 'FullSim' or dofastfull == True): referenceSample=RefRepository+'/'+RefRelease+'/'+RefSelection+'/'+sample+'/'+'val.'+sample+'.root'
            if os.path.isfile(referenceSample ):
                    replace_map = { 'NEW_FILE':'val.'+sample+'.root', 'REF_FILE':RefRelease+'/'+RefSelection+'/val.'+sample+'.root', 'REF_LABEL':sample, 'NEW_LABEL': sample, 'REF_RELEASE':RefRelease, 'NEW_RELEASE':NewRelease, 'REFSELECTION':RefSelection, 'NEWSELECTION':NewSelection, 'TrackValHistoPublisher': cfgFileName, 'MINEFF':mineff, 'MAXEFF':maxeff, 'MAXFAKE':maxfake}

                    if(os.path.exists(RefRelease+'/'+RefSelection)==False):
                            os.makedirs(RefRelease+'/'+RefSelection)
                    os.system('cp ' + referenceSample+ ' '+RefRelease+'/'+RefSelection)  
            else:
                    print "No reference file found at: ", RefRelease+'/'+RefSelection
                    replace_map = { 'NEW_FILE':'val.'+sample+'.root', 'REF_FILE':'val.'+sample+'.root', 'REF_LABEL':sample, 'NEW_LABEL': sample, 'REF_RELEASE':NewRelease, 'NEW_RELEASE':NewRelease, 'REFSELECTION':NewSelection, 'NEWSELECTION':NewSelection, 'TrackValHistoPublisher': cfgFileName, 'MINEFF':mineff, 'MAXEFF':maxeff, 'MAXFAKE':maxfake}


            macroFile = open(cfgFileName+'.C' , 'w' )
            replace(replace_map, templatemacroFile, macroFile)


            os.system('root -b -q -l '+ cfgFileName+'.C'+ '>  macro.'+cfgFileName+'.log')


            if(os.path.exists(newdir)==False):
                    os.makedirs(newdir)

            print "moving pdf files for sample: " , sample
            os.system('mv  *.pdf ' + newdir)

            print "moving root file for sample: " , sample
            os.system('mv val.'+ sample+ '.root ' + newdir)

            print "copy py file for sample: " , sample
            if (Sequence!="comparison_only"): 
                os.system('cp '+cfgFileName+'.py ' + newdir)
	
	
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

NewRepositoryBase = copy.copy(NewRepository)
for algo in Algos:
    for quality in Qualities:
        NewRepository = copy.copy(NewRepositoryBase)
        dofastfull = False
        PileUp = 'noPU'
        sampleType = 'FullSim'
        RefSelection=RefStartupTag+'_'+PileUp
        if( quality !=''):
            RefSelection+='_'+quality
        if(algo!=''and not(algo=='ootb' and quality !='')):
            RefSelection+='_'+algo
        if(quality =='') and (algo==''):
            RefSelection+='_ootb'
        do_validation(startupsamples, StartupTag, quality , algo, PileUp, sampleType, dofastfull)

        NewRepository = copy.copy(NewRepositoryBase)
        PileUp = 'PU'
        sampleType = 'FullSim'
        RefSelection=RefStartupTag+'_'+PileUp
        if( quality !=''):
            RefSelection+='_'+quality
        if(algo!=''and not(algo=='ootb' and quality !='')):
            RefSelection+='_'+algo
        if(quality =='') and (algo==''):
            RefSelection+='_ootb'
        do_validation(pileupstartupsamples, StartupTag, quality , algo, PileUp, sampleType, dofastfull)

        NewRepository = copy.copy(NewRepositoryBase) + '/fastsim'
        PileUp = 'noPU'
        sampleType = 'FastSim'
        RefSelection=RefStartupTag+'_'+PileUp
        if( quality !=''):
            RefSelection+='_'+quality
        if(algo!=''and not(algo=='ootb' and quality !='')):
            RefSelection+='_'+algo
        if(quality =='') and (algo==''):
            RefSelection+='_ootb'
        do_validation(fastsimstartupsamples, StartupTag, quality , algo, PileUp, sampleType, dofastfull)
        
        NewRepository = copy.copy(NewRepositoryBase) + '/fastsim'
        PileUp = 'PU'
        sampleType = 'FastSim'
        RefSelection=RefStartupTag+'_'+PileUp
        if( quality !=''):
            RefSelection+='_'+quality
        if(algo!=''and not(algo=='ootb' and quality !='')):
            RefSelection+='_'+algo
        if(quality =='') and (algo==''):
            RefSelection+='_ootb'
        do_validation(pileupfastsimstartupsamples, StartupTag, quality , algo, PileUp, sampleType, dofastfull)

        NewRepository = copy.copy(NewRepositoryBase) + '/fastfull'
        dofastfull = True
        PileUp = 'noPU'
        sampleType = 'FastSim'
        RefSelection=RefStartupTag+'_'+PileUp
        if( quality !=''):
            RefSelection+='_'+quality
        if(algo!=''and not(algo=='ootb' and quality !='')):
            RefSelection+='_'+algo
        if(quality =='') and (algo==''):
            RefSelection+='_ootb'
        do_validation(fastsimstartupsamples, StartupTag, quality , algo, PileUp, sampleType, dofastfull)

        NewRepository = copy.copy(NewRepositoryBase) + '/fastfull'
        PileUp = 'PU'
        sampleType = 'FastSim'
        dofastfull = True
        RefSelection=RefStartupTag+'_'+PileUp
        if( quality !=''):
            RefSelection+='_'+quality
        if(algo!=''and not(algo=='ootb' and quality !='')):
            RefSelection+='_'+algo
        if(quality =='') and (algo==''):
            RefSelection+='_ootb'
        do_validation(pileupfastsimstartupsamples, StartupTag, quality , algo, PileUp, sampleType, dofastfull)

