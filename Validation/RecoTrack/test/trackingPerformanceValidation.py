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
NewRelease='CMSSW_3_1_2'

# startup and ideal sample list

#This are the standard relvals (startup)
startupsamples= ['RelValTTbar', 'RelValMinBias', 'RelValQCD_Pt_3000_3500']

#This is pileup sample
#startupsamples= ['RelValTTbar_Tauola']

#to skip startup samples:
#startupsamples= []

#This are the standard relvals (ideal)
idealsamples= ['RelValSingleMuPt1', 'RelValSingleMuPt10', 'RelValSingleMuPt100', 'RelValSinglePiPt1', 'RelValSinglePiPt10', 'RelValSinglePiPt100', 'RelValSingleElectronPt35', 'RelValTTbar', 'RelValQCD_Pt_3000_3500','RelValMinBias']

#This is pileup sample
#idealsamples= ['RelValZmumuJets_Pt_20_300_GEN']

#summer09 preproduction (the character '-' must be avoided)
#idealsamples= ['InclusiveMu5_Pt250__Summer09', 'InclusiveMu5_Pt50__Summer09', 'MinBias_herwig__Summer09', 'TTbar__Summer09']

#to skip ideal samples:
#idealsamples= []



# track algorithm name and quality. Can be a list.
#Algos= ['ootb']
Algos= ['ootb', 'iter0', 'iter1','iter2','iter3','iter4','iter5']
#Qualities=['']
Qualities=['', 'highPurity']

#Leave unchanged unless the track collection name changes
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

Sequence='preproduction'
#Sequence='harvesting'


# Ideal and Statup tags
IdealTag='MC_31X_V3'
StartupTag='STARTUP31X_V2'

# PileUp: PU . No PileUp: noPU
PileUp='noPU'

# Reference directory name (the macro will search for ReferenceSelection_Quality_Algo)
ReferenceSelection='MC_31X_V2_'+PileUp
StartupReferenceSelection='STARTUP31X_V1_'+PileUp

# Default label is GlobalTag_noPU__Quality_Algo. Change this variable if you want to append an additional string.
NewSelectionLabel=''
#NewSelectionLabel='test2_logpt'


#Reference and new repository
RefRepository = '/afs/cern.ch/cms/performance/tracker/activities/reconstruction/tracking_performance'
NewRepository = '/afs/cern.ch/cms/performance/tracker/activities/reconstruction/tracking_performance'

#for preproduction samples:
#RefRepository = '/afs/cern.ch/cms/performance/tracker/activities/reconstruction/tracking_performance/preproduction'
#NewRepository = '/afs/cern.ch/cms/performance/tracker/activities/reconstruction/tracking_performance/preproduction'

#Default Nevents
defaultNevents ='-1'

#Put here the number of event to be processed for specific samples (numbers must be strings) if not specified is defaultNevents:
Events={}
#Events={'RelValTTbar':'100'}

# template file names. Usually should not be changed.
cfg='trackingPerformanceValidation_cfg.py'
macro='macro/TrackValHistoPublisher.C'



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

    
def do_validation(samples, GlobalTag, trackquality, trackalgorithm):
    global Sequence, RefSelection, RefRepository, NewSelection, NewRepository, defaultNevents, Events
    global cfg, macro, Tracksname
    print 'Tag: ' + GlobalTag
    tracks_map = { 'ootb':'general_AssociatorByHitsRecoDenom','iter0':'cutsRecoZero_AssociatorByHitsRecoDenom','iter1':'cutsRecoFirst_AssociatorByHitsRecoDenom','iter2':'cutsRecoSecond_AssociatorByHitsRecoDenom','iter3':'cutsRecoThird_AssociatorByHitsRecoDenom','iter4':'cutsRecoFourth_AssociatorByHitsRecoDenom','iter5':'cutsRecoFifth_AssociatorByHitsRecoDenom'}
    tracks_map_hp = { 'ootb':'cutsRecoHp_AssociatorByHitsRecoDenom','iter0':'cutsRecoZeroHp_AssociatorByHitsRecoDenom','iter1':'cutsRecoFirstHp_AssociatorByHitsRecoDenom','iter2':'cutsRecoSecondHp_AssociatorByHitsRecoDenom','iter3':'cutsRecoThirdHp_AssociatorByHitsRecoDenom','iter4':'cutsRecoFourthHp_AssociatorByHitsRecoDenom','iter5':'cutsRecoFifthHp_AssociatorByHitsRecoDenom'}
    if(trackalgorithm=='iter0' or trackalgorithm=='ootb'):
        mineff='0.5'
        maxeff='1.025'
        maxfake='0.7'
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

    #loop on all the requested samples
    for sample in samples :
        templatecfgFile = open(cfg, 'r')
        templatemacroFile = open(macro, 'r')
        print 'Get information from DBS for sample', sample
        newdir=NewRepository+'/'+NewRelease+'/'+NewSelection+'/'+sample 
	cfgFileName=sample+GlobalTag
        #check if the sample is already done
        if(os.path.isfile(newdir+'/building.pdf' )!=True):    
            #if the job is harvesting check if the file is already harvested
            if( Sequence=="harvesting"):
            	harvestedfile='./DQM_V0001_R000000001__' + GlobalTag+ '__' + sample + '__Validation.root'
            elif( Sequence=="preproduction"):
                harvestedfile='./DQM_V0001_R000000001__' + sample+ '-' + GlobalTag + '_preproduction_312-v1__GEN-SIM-RECO_1.root'
		if(( (Sequence=="harvesting" or Sequence=="preproduction") and os.path.isfile(harvestedfile) )==False):
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
				filenames+=']);\n'
				
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
		
				cmdrun='cmsRun ' +cfgFileName+ '.py >&  ' + cfgFileName + '.log < /dev/zero '
				retcode=os.system(cmdrun)
		
			else:      
				print 'No dataset found skipping sample: '+ sample, '\n'  
				continue
		else: 
			retcode=0
		if (retcode!=0):
			print 'Job for sample '+ sample + ' failed. \n'
		else:
			if (Sequence=="harvesting" or Sequence=="preproduction"):
				#copy only the needed histograms
				if(trackquality==""):
					rootcommand='root -b -q -l CopySubdir.C\\('+ '\\\"'+harvestedfile+'\\\",\\\"val.' +sample+'.root\\\",\\\"'+ tracks_map[trackalgorithm]+ '\\\"\\) >& /dev/null'
					os.system(rootcommand)
				elif(trackquality=="highPurity"):
					os.system('root -b -q -l CopySubdir.C\\('+ '\\\"'+harvestedfile+'\\\",\\\"val.' +sample+'.root\\\",\\\"'+ tracks_map_hp[trackalgorithm]+ '\\\"\\) >& /dev/null')
		
		
			referenceSample=RefRepository+'/'+RefRelease+'/'+RefSelection+'/'+sample+'/'+'val.'+sample+'.root'
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

for algo in Algos:
    for quality in Qualities:
        RefSelection=ReferenceSelection
        if( quality !=''):
            RefSelection+='_'+quality
        if(algo!=''and not(algo=='ootb' and quality !='')):
            RefSelection+='_'+algo
        if(quality =='') and (algo==''):
            RefSelection+='_ootb'
        do_validation(idealsamples, IdealTag, quality , algo)
        RefSelection=StartupReferenceSelection
        if( quality !=''):
            RefSelection+='_'+quality
        if(algo!=''and not(algo=='ootb' and quality !='')):
            RefSelection+='_'+algo
        if(quality =='') and (algo==''):
            RefSelection+='_ootb'
        do_validation(startupsamples, StartupTag, quality , algo)
        
