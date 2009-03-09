#! /usr/bin/env python

import os
import sys
import fileinput
import string

##########################################################################
##########################################################################
######### User variables

#Reference release

RefRelease='CMSSW_3_1_0_pre2'

# startup and ideal sample list
startupsamples= ['RelValTTbar', 'RelValMinBias', 'RelValQCD_Pt_3000_3500']
#startupsamples= ['RelValTTbar']

idealsamples= ['RelValSingleMuPt1', 'RelValSingleMuPt10', 'RelValSingleMuPt100', 'RelValSinglePiPt1', 'RelValSinglePiPt10', 'RelValSinglePiPt100', 'RelValSingleElectronPt35', 'RelValTTbar', 'RelValQCD_Pt_3000_3500','RelValMinBias']

#idealsamples= [ 'RelValSingleElectronPt35']
#idealsamples= ['RelValTTbar']



# track algorithm name and quality. Can be a list.
Algos= ['ootb', 'ctf','iter2','iter3','iter4','iter5']
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

#Sequence='only_validation_and_TP'
Sequence='harvesting'

# Ideal and Statup tags
IdealTag='IDEAL_30X'
StartupTag='STARTUP_30X'

# Reference directory name (the macro will search for ReferenceSelection_Quality_Algo)
ReferenceSelection='IDEAL_30X_noPU'
StartupReferenceSelection='STARTUP_30X_noPU'

# Default label is GlobalTag_noPU__Quality_Algo. Change this variable if you want to append an additional string.
NewSelectionLabel=''


#Reference and new repository
RefRepository = '/afs/cern.ch/cms/performance/tracker/activities/reconstruction/tracking_performance'
NewRepository = '/afs/cern.ch/cms/performance/tracker/activities/reconstruction/tracking_performance'

#Default Nevents
defaultNevents ='-1'

#Put here the number of event to be processed for specific samples (numbers must be strings) if not specified is -1:
Events={ 'RelValQCD_Pt_3000_3500':'5000', 'RelValTTbar':'5000', 'RelValQCD_Pt_80_120':'5000', 'RelValBJets_Pt_50_120':'5000'}

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
    tracks_map = { 'ootb':'general_AssociatorByHits','ctf':'cutsRecoFirst_AssociatorByHits','iter2':'cutsRecoSecond_AssociatorByHits','iter3':'cutsRecoThird_AssociatorByHits','iter4':'cutsRecoFourth_AssociatorByHits','iter5':'cutsRecoFifth_AssociatorByHits'}
    tracks_map_hp = { 'ootb':'cutsRecoHp_AssociatorByHits','ctf':'cutsRecoFirstHp_AssociatorByHits','iter2':'cutsRecoSecondHp_AssociatorByHits','iter3':'cutsRecoThirdHp_AssociatorByHits','iter4':'cutsRecoFourthHp_AssociatorByHits','iter5':'cutsRecoFifthHp_AssociatorByHits'}
    if(trackalgorithm=='ctf' or trackalgorithm=='ootb'):
        mineff='0.5'
        maxeff='1.025'
        maxfake='0.7'
    else:
        mineff='0'
        maxeff='0.1'
        maxfake='0.8'
    #build the New Selection name
    NewSelection=GlobalTag +'_noPU'
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
	cfgFileName=sample
        #check if the sample is already done
        if(os.path.isfile(newdir+'/building.pdf' )!=True):    
            #if the job is harvesting check if the file is already harvested
            	harvestedfile='./DQM_V0001_R000000001__' + GlobalTag+ '__' + sample + '__Validation.root'
		if(( Sequence=="harvesting" and os.path.isfile(harvestedfile) )==False):
			#search the primary dataset
			cmd='./DDSearchCLI.py  --limit -1 --input="find  dataset.createdate, dataset where dataset like *'
		#            cmd+=sample+'/'+NewRelease+'_'+GlobalTag+'*GEN-SIM-DIGI-RAW-HLTDEBUG-RECO* "'
			cmd+=sample+'/'+NewRelease+'_'+GlobalTag+'*GEN-SIM-RECO* "'
			cmd+='|grep '+sample+'|grep -v test|sort|tail -1| cut -d "," -f2 '
			print cmd
			dataset= os.popen(cmd).readline()
			print 'DataSet:  ', dataset, '\n'
		
			#Check if a dataset is found
			if dataset!="":
		
				#Find and format the list of files
				cmd2='./DDSearchCLI.py  --limit -1 --cff --input="find file where dataset like '+ dataset +'"|grep ' + sample 
				filenames='import FWCore.ParameterSet.Config as cms\n'
				filenames+='readFiles = cms.untracked.vstring()\n'
				filenames+='secFiles = cms.untracked.vstring()\n'
				filenames+='source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)\n'
				filenames+='readFiles.extend( [\n'
				for filename in os.popen(cmd2).readlines():
					filenames+=filename
				filenames+=']);\n'
				
				# if not harvesting find secondary file names
				if(Sequence!="harvesting"):
					cmd3='./DDSearchCLI.py  --limit -1 --input="find dataset.parent where dataset like '+ dataset +'"|grep ' + sample
					parentdataset=os.popen(cmd3).readline()
					print 'Parent DataSet:  ', parentdataset, '\n'
		
				#Check if a dataset is found
					if parentdataset!="":
						cmd4='./DDSearchCLI.py  --limit -1 --cff --input="find file where dataset like '+ parentdataset +'"|grep ' + sample 
						filenames+='secFiles.extend( [\n'
						first=True
		
						for line in os.popen(cmd4).readlines():
							filenames+=line
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
		
				symbol_map = { 'NEVENT':Nevents, 'GLOBALTAG':GlobalTag, 'SEQUENCE':Sequence, 'SAMPLE': sample, 'ALGORITHM':trackalgorithm, 'QUALITY':trackquality, 'TRACKS':Tracks}
		
		
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
			if Sequence=="harvesting":
				#copy only the needed histograms
				if(trackquality==""):
					rootcommand='root -b -q -l CopySubdir.C\\('+ '\\\"DQM_V0001_R000000001__' + GlobalTag+ '__' + sample + '__Validation.root\\\",\\\"val.' +sample+'.root\\\",\\\"'+ tracks_map[trackalgorithm]+ '\\\"\\) >& /dev/null'
					os.system(rootcommand)
				elif(trackquality=="highPurity"):
					os.system('root -b -q -l CopySubdir.C\\('+ '\\\"DQM_V0001_R000000001__' + GlobalTag+ '__' + sample + '__Validation.root\\\",\\\"val.' +sample+'.root\\\",\\\"'+ tracks_map_hp[trackalgorithm]+ '\\\"\\) >& /dev/null')
		
		
			referenceSample=RefRepository+'/'+RefRelease+'/'+RefSelection+'/'+sample+'/'+'val.'+sample+'.root'
			if os.path.isfile(referenceSample ):
				replace_map = { 'NEW_FILE':'val.'+sample+'.root', 'REF_FILE':RefRelease+'/'+RefSelection+'/val.'+sample+'.root', 'REF_LABEL':sample, 'NEW_LABEL': sample, 'REF_RELEASE':RefRelease, 'NEW_RELEASE':NewRelease, 'REFSELECTION':RefSelection, 'NEWSELECTION':NewSelection, 'TrackValHistoPublisher': sample, 'MINEFF':mineff, 'MAXEFF':maxeff, 'MAXFAKE':maxfake}
		
				if(os.path.exists(RefRelease+'/'+RefSelection)==False):
					os.makedirs(RefRelease+'/'+RefSelection)
				os.system('cp ' + referenceSample+ ' '+RefRelease+'/'+RefSelection)  
			else:
				print "No reference file found at: ", RefRelease+'/'+RefSelection
                                replace_map = { 'NEW_FILE':'val.'+sample+'.root', 'REF_FILE':'val.'+sample+'.root', 'REF_LABEL':sample, 'NEW_LABEL': sample, 'REF_RELEASE':NewRelease, 'NEW_RELEASE':NewRelease, 'REFSELECTION':NewSelection, 'NEWSELECTION':NewSelection, 'TrackValHistoPublisher': sample, 'MINEFF':mineff, 'MAXEFF':maxeff, 'MAXFAKE':maxfake}
		
		
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
            
try:
     #Get some environment variables to use
     NewRelease     = os.environ["CMSSW_VERSION"]

except KeyError:
     print >>sys.stderr, 'Error: The environment variable CMSSW_VERSION is not available.'
     print >>sys.stderr, '       Please run eval `scramv1 runtime -csh` to set your environment variables'
     sys.exit()


#print 'Running validation on the following samples: ', samples

if os.path.isfile( 'DDSearchCLI.py')!=True:
    e =os.system("wget --no-check-certificate https://cmsweb.cern.ch/dbs_discovery/aSearchCLI -O DDSearchCLI.py")
    if  e != 0:
        print >>sys.stderr, "Failed to dowload dbs aSearch file (https://cmsweb.cern.ch/dbs_discovery/aSearchCLI)"
        print >>sys.stderr, "Child was terminated by signal", e
        os.remove('DDSearchCLI.py')
        sys.exit()
    else:
        os.system('chmod +x DDSearchCLI.py')

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
        
