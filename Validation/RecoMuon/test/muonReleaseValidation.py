#! /usr/bin/env python

import os
import sys
import fileinput
import string

##########################################################################
##########################################################################
######### User variables

#Reference release

RefRelease='CMSSW_2_2_3'

# startup and ideal sample list
#startupsamples= ['RelValTTbar', 'RelValMinBias', 'RelValQCD_Pt_3000_3500']
#startupsamples= ['RelValTTbar']
startupsamples= ['']

#idealsamples= ['RelValSingleMuPt1', 'RelValSingleMuPt10', 'RelValSingleMuPt100', 'RelValSinglePiPt1', 'RelValSinglePiPt10', 'RelValSinglePiPt100', 'RelValSingleElectronPt35', 'RelValTTbar', 'RelValQCD_Pt_3000_3500','RelValMinBias']

#idealsamples= [ 'RelValSingleElectronPt35']
idealsamples= ['RelValTTbar']



# track algorithm name and quality. Can be a list.
Algos= ['']
Qualities=['']
#Qualities=['', 'highPurity']

#Leave unchanged unless the track collection name changed
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
#Sequence='report_only'

Submit=True
DBS=True
Publish=False

# Ideal and Statup tags
IdealTag='IDEAL_30X'
StartupTag='STARTUP_V7'

# Reference directory name (the macro will search for ReferenceSelection_Quality_Algo)
ReferenceSelection='IDEAL_V11_noPU'
StartupReferenceSelection='STARTUP_V7_noPU'

# Default label is GlobalTag_noPU__Quality_Algo. Change this variable if you want to append an additional string.
NewSelectionLabel=''


#Reference and new repository
RefRepository = '/afs/cern.ch/cms/Physics/muon/CMSSW/Performance/RecoMuon/Validation/data'
NewRepository = '/afs/cern.ch/cms/Physics/muon/CMSSW/Performance/RecoMuon/Validation/data'

#Default Nevents
defaultNevents ='-1'

#Put here the number of event to be processed for specific samples (numbers must be strings) if not specified is -1:
Events={ 'RelValQCD_Pt_3000_3500':'5000', 'RelValTTbar':'5000', 'RelValQCD_Pt_80_120':'5000', 'RelValBJets_Pt_50_120':'5000'}

# template file names. Usually should not be changed.
cfg='muonReleaseValidation_cfg.py'
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
        templatecfgFile = open(cfg, 'r')
        templatemacroFile = open(macro, 'r')
        print 'Get information from DBS for sample', sample
        newdir=NewRepository+'/'+NewRelease+'/'+NewSelection+'/'+sample 
        
        #chech if the sample is already done
        if(os.path.isfile(newdir+'/building.pdf' )!=True):    
            
            #search the primary dataset
            cmd='./DDSearchCLI.py  --limit -1 --input="find  dataset.createdate, dataset where dataset like *'
            #            cmd+=sample+'/'+NewRelease+'_'+GlobalTag+'*GEN-SIM-DIGI-RAW-HLTDEBUG-RECO* "'
            cmd+=sample+'/'+NewRelease+'_'+GlobalTag+'*GEN-SIM-RECO* "'
            cmd+='|grep '+sample+'|sort|tail -1| cut -d "," -f2 '
            print cmd
            dataset= os.popen(cmd).readline()
            print 'DataSet:  ', dataset, '\n'
            
            #Check if a dataset is found
            if dataset!="":
                print 'dataset found'
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
                print 'made many names'
                # if not harvesting find secondary file names
                if(Sequence!="harvesting" and Sequence!="report_only"):
                    print 'NOT HARVESTING'
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
#                            secfilename=line.strip()
#                            if first==True:
#                                filenames+="'"
#                                first=False
#                            else :
#                                filenames+=",\n'"
#                            filenames+=secfilename
#                            filenames+="'"

                        filenames+=']);\n'
                    else :
                        print "No primary dataset found skipping sample: ", sample
                        continue
                else :
                    filenames+='secFiles.extend( (               ) )'
                    print 'ELSE HARVESTING'
                cfgFileName=sample
                print 'cfgFileName ' + cfgFileName
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

                if(os.path.exists(NewRelease+'/'+NewSelection+'/'+sample)==False):
                    os.makedirs(NewRelease+'/'+NewSelection+'/'+sample)

                retcode=0
                if(Sequence!="report_only"):
                    if(Submit):
                        retcode=os.system(cmdrun)
                    else:
                        sys.exit()
                else:
                    newSample=NewRepository+'/'+NewRelease+'/'+NewSelection+'/'+sample+'/'+'val.'+sample+'.root'
                    if os.path.isfile(newSample ):
                        os.system('cp ' + newSample+ ' '+NewRelease+'/'+NewSelection+'/'+sample)  
                    else:
                        print "No new file found at: ", NewRelease+'/'+NewSelection+'/'+sample

                if (retcode!=0):
                    print 'Job for sample '+ sample + ' failed. \n'
                else:
                    if Sequence=="harvesting":
                        os.system('mv  DQM_V0001_R000000001__' + GlobalTag+ '__' + sample + '__Validation.root' + ' ' + NewRelease+'/'+NewSelection+'/'+sample+'/val.' +sample+'.root')
                    else:
                        os.system('mv  val.' + sample + '.root' + ' ' + NewRelease+'/'+NewSelection+'/'+sample+'/val.' +sample+'.root')
                    referenceSample=RefRepository+'/'+RefRelease+'/'+RefSelection+'/'+sample+'/'+'val.'+sample+'.root'
                    if os.path.isfile(referenceSample ):
                        replace_map = { 'NEW_FILE':NewRelease+'/'+NewSelection+'/'+sample+'/val.'+sample+'.root', 'REF_FILE':RefRelease+'/'+RefSelection+'/val.'+sample+'.root', 'REF_LABEL':sample, 'NEW_LABEL': sample, 'REF_RELEASE':RefRelease, 'NEW_RELEASE':NewRelease, 'REFSELECTION':RefSelection, 'NEWSELECTION':NewSelection, 'TrackValHistoPublisher': sample}

                        if(os.path.exists(RefRelease+'/'+RefSelection)==False):
                            os.makedirs(RefRelease+'/'+RefSelection)
                        os.system('cp ' + referenceSample+ ' '+RefRelease+'/'+RefSelection)  
                    else:
                        print "No reference file found at: ", RefRelease+'/'+RefSelection
                        replace_map = { 'NEW_FILE':NewRelease+'/'+NewSelection+'/'+sample+'/val.'+sample+'.root', 'REF_FILE':NewRelease+'/'+NewSelection+'/val.'+sample+'.root', 'REF_LABEL':sample, 'NEW_LABEL': sample, 'REF_RELEASE':NewRelease, 'NEW_RELEASE':NewRelease, 'REFSELECTION':NewSelection, 'NEWSELECTION':NewSelection, 'TrackValHistoPublisher': sample}


                    macroFile = open(cfgFileName+'.C' , 'w' )
                    replace(replace_map, templatemacroFile, macroFile)


                    os.system('root -b -q -l '+ cfgFileName+'.C'+ '>  macro.'+cfgFileName+'.log')


                    if(os.path.exists(newdir)==False):
                        os.makedirs(newdir)

                    if(Publish):
                        print "copying pdf files for sample: " , sample
                        os.system('scp -r '+NewRelease+'/'+NewSelection+'/'+sample+'/* ' + newdir)

                    if(Sequence!='report_only' and Publish):
                        print "copying root file for sample: " , sample
                        os.system('cp '+NewRelease+'/'+NewSelection+'/'+sample+'/val.'+ sample+ '.root ' + newdir)
                        print "copying py file for sample: " , sample
                        os.system('cp '+cfgFileName+'.py ' + newdir)

            elif DBS==False:
                filenames='import FWCore.ParameterSet.Config as cms\n'
                filenames+='readFiles = cms.untracked.vstring()\n'
                filenames+='secFiles = cms.untracked.vstring()\n'
                filenames+='source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)\n'
                filenames+='readFiles.extend( [\n'
                filenames+=']);\n'
                cfgFileName=sample
                print 'cfgFileName ' + cfgFileName
                cfgFile = open(cfgFileName+'.py' , 'w' )
                cfgFile.write(filenames)
                if (Events.has_key(sample)!=True):
                    Nevents=defaultNevents
                else:
                    Nevents=Events[sample]
                    
                symbol_map = { 'NEVENT':Nevents, 'GLOBALTAG':GlobalTag, 'SEQUENCE':Sequence, 'SAMPLE': sample, 'ALGORITHM':trackalgorithm, 'QUALITY':trackquality, 'TRACKS':Tracks}


                cfgFile = open(cfgFileName+'.py' , 'a' )
                replace(symbol_map, templatecfgFile, cfgFile)
            else:
                print 'No dataset found skipping sample: '+ sample, '\n'
        else:
            print 'Validation for sample ' + sample + ' already done. Skipping this sample. \n'




##########################################################################
#########################################################################
######## This is the main program
            
try:
     #Get some environment variables to use
     NewRelease     = os.environ["CMSSW_VERSION"]
#      NewRelease='CMSSW_3_0_0_pre6'
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
    print 'algo ' + algo
    for quality in Qualities:
        print 'quality ' + quality
        RefSelection=ReferenceSelection
        print 'Before RefSelection: ' + RefSelection
        if( quality !=''):
            RefSelection+='_'+quality
        if(algo!=''):
            RefSelection+='_'+algo
        if(quality =='') and (algo==''):
            RefSelection+='_ootb'
        print 'After RefSelection: ' + RefSelection
        do_validation(idealsamples, IdealTag, quality , algo)
        RefSelection=StartupReferenceSelection
        print 'Before StartupRefSelection: ' + RefSelection
        if( quality !=''):
            RefSelection+='_'+quality
        if(algo!=''):
            RefSelection+='_'+algo
        if(quality =='') and (algo==''):
            RefSelection+='_ootb'
        print 'After StartupRefSelection: ' + RefSelection
        do_validation(startupsamples, StartupTag, quality , algo)

