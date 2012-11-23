#! /usr/bin/env python

import os
import sys
import fileinput
import string

#NewRelease='CMSSW_6_0_1_PostLS1v2'
NewRelease='CMSSW_6_1_0_pre6'
RefRelease='CMSSW_6_1_0_pre5'

#NewCondition='MC'
#RefCondition='MC'
NewCondition='STARTUP'
RefCondition='STARTUP'
#NewCondition='PILEUP'
#RefCondition='PILEUP'
#NewCondition='POSTLS1'
#RefCondition='POSTLS1'
theGuiPostFixLS1 = "_UPGpostls1_14"

NewFastSim=False
RefFastSim=False

if (NewCondition=='MC'):
    samples= ['RelValSingleMuPt10','RelValSingleMuPt100','RelValSingleMuPt1000','RelValTTbar']
    if (NewFastSim|RefFastSim):
        samples= ['RelValSingleMuPt10','RelValSingleMuPt100','RelValTTbar']
elif (NewCondition=='STARTUP'):
    samples= ['RelValSingleMuPt10','RelValSingleMuPt100','RelValSingleMuPt1000','RelValTTbar','RelValZMM','RelValJpsiMM']
    if (NewFastSim|RefFastSim):
        samples= ['RelValSingleMuPt10','RelValSingleMuPt100','RelValTTbar']
if ((NewCondition=="POSTLS1")|(RefCondition=="POSTLS1")):
    samples= ['RelValZMM','RelValJpsiMM']
if ((NewCondition=='PILEUP')|(RefCondition=='PILEUP')):
    samples= ['RelValTTbar']
    if (NewFastSim|RefFastSim):
        samples= ['RelValTTbar']

Submit=True
Publish=False

# Where to get the root file from.
# By default, if the root files are already in the local area, they won't be overwritten

#GetFilesFrom='WEB'       # --> Take root files from the MuonPOG Validation repository on the web
#GetFilesFrom='CASTOR'    # --> Copy root files from castor
GetFilesFrom='GUI'       # --> Copy root files from the DQM GUI server
#GetRefsFrom='WEB'
#GetRefsFrom='CASTOR'
GetRefsFrom='GUI'

#DqmGuiNewRepository = 'https://cmsweb.cern.ch/dqm/dev/data/browse/Development/RelVal/CMSSW_4_2_x/'
#DqmGuiNewRepository = 'https://cmsweb.cern.ch/dqm/offline/data/browse/ROOT/RelVal/CMSSW_4_3_x/'
DqmGuiNewRepository = 'https://cmsweb.cern.ch/dqm/relval/data/browse/ROOT/RelVal/CMSSW_6_1_x/'
#DqmGuiRefRepository = 'https://cmsweb.cern.ch/dqm/dev/data/browse/Development/RelVal/CMSSW_4_2_x/'
#DqmGuiRefRepository = 'https://cmsweb.cern.ch/dqm/offline/data/browse/ROOT/RelVal/CMSSW_4_3_x/'
DqmGuiRefRepository = 'https://cmsweb.cern.ch/dqm/relval/data/browse/ROOT/RelVal/CMSSW_6_1_x/'
CastorRepository = '/castor/cern.ch/user/a/aperrott/ValidationRecoMuon'

# These are only needed if you copy any root file from the DQM GUI:
NewLabel='START61_V5'
if (NewCondition=='MC'):
    NewLabel='MC_52_V1'
elif (NewCondition=='POSTLS1'):
    NewLabel='POSTLS161_V12'
    
RefLabel='START61_V4'
if (RefCondition=='MC'):
    RefLabel='MC_52_V1'
elif (RefCondition=='POSTLS1'):
    RefLabel='POSTLS161_V12'

if ((GetFilesFrom=='GUI')|(GetRefsFrom=='GUI')):
    print "*** Did you remind doing:"
#    print " > source /afs/cern.ch/cms/LCG/LCG-2/UI/cms_ui_env.csh"
    print " > source /afs/cern.ch/project/gd/LCG-share/sl5/etc/profile.d/grid_env.csh"
    print " > voms-proxy-init"


ValidateHLT=True
ValidateRECO=True
ValidateISO=True
ValidateDQM=True
if (NewFastSim|RefFastSim):
    ValidateDQM=False
#    ValidateRECO=False


#NewFormat='GEN-SIM-RECO'
#RefFormat='GEN-SIM-RECO'
NewFormat='DQM'
RefFormat='DQM'
NewTag = NewCondition+'_noPU'
RefTag = RefCondition+'_noPU'
if (NewCondition=='PILEUP'):
    NewTag = NewCondition+'_PU'
if (RefCondition=='PILEUP'):
    RefTag = RefCondition+'_PU'

#specify if any of the files compared is from FastSim
isFastSimNew = ''
isFastSimOld = ''
if (NewFastSim):
    isFastSimNew = 'FS'
    NewTag = NewTag+'_FSIM'
    NewLabel=NewLabel+'_FastSim'
    NewFormat='GEN-SIM-DIGI-RECO'    
if (RefFastSim):
    isFastSimOld = 'FS'
    RefTag = RefTag+'_FSIM'
    RefLabel=RefLabel+'_FastSim'
    RefFormat='GEN-SIM-DIGI-RECO'

if (NewCondition=='PILEUP'):
    if (NewFastSim):
        NewLabel='PU_'+NewLabel
    else:
        NewLabel='PU_'+NewLabel
if (RefCondition=='PILEUP'):
    if (RefFastSim):
        RefLabel='PU_'+RefLabel
    else:
        RefLabel='PU_'+RefLabel

NewLabel=NewLabel+'-v1'
RefLabel=RefLabel+'-v1'


WebRepository = '/afs/cern.ch/cms/Physics/muon/CMSSW/Performance/RecoMuon/Validation/val'
CastorRefRepository = '/castor/cern.ch/user/a/aperrott/ValidationRecoMuon'    

#specify macros used to plot here
macro='macro/TrackValHistoPublisher.C'
macroSeed='macro/SeedValHistoPublisher.C'
macroReco='macro/RecoValHistoPublisher.C'
macroIsol='macro/IsoValHistoPublisher.C'
macroMuonReco='macro/RecoMuonValHistoPublisher.C'

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

###############################

for sample in samples :

    if(os.path.exists(NewRelease+'/'+NewTag+'/'+sample)==False):
        os.makedirs(NewRelease+'/'+NewTag+'/'+sample)
    if(os.path.exists(RefRelease+'/'+RefTag+'/'+sample)==False):
        os.makedirs(RefRelease+'/'+RefTag+'/'+sample)

    checkFile = NewRelease+'/'+NewTag+'/'+sample+'/general_tpToTkmuAssociation.pdf'
    if (RefFastSim):
        checkFile = NewRelease+'/'+NewTag+'/'+sample+'/general_tpToTkmuAssociationFS.pdf'
    if (os.path.isfile(checkFile)==True):
        print "Files of type "+checkFile+' exist alredy: delete them first, if you really want to overwrite them'
    else:
        newSampleOnWeb=WebRepository+'/'+NewRelease+'/'+NewTag+'/'+sample+'/'+'val.'+sample+'.root'
        refSampleOnWeb=WebRepository+'/'+RefRelease+'/'+RefTag+'/'+sample+'/'+'val.'+sample+'.root'

        if (os.path.isfile(NewRelease+'/'+NewTag+'/'+sample+'/val.'+sample+'.root')==True):
            print "New file found at: "+NewRelease+'/'+NewTag+'/'+sample+'/val.'+sample+'.root'+' -> Use that one'
        elif (GetFilesFrom=='GUI'):
            theGuiSample = sample
            # Temporary fix due to the wrong name used for JPsiMM in the DQM GUI
            if ((sample=="RelValJpsiMM")&(NewCondition=='POSTLS1')):
                theGuiSample = "RelValJpsiMMM"
            #
            if (NewCondition=='POSTLS1'):
                theGuiSample = theGuiSample+theGuiPostFixLS1
#            newGuiFileName='DQM_V0001_R000000001__'+sample+'__'+NewRelease+'-'+NewLabel+'__'+NewFormat+'.root '
            newGuiFileName='DQM_V0001_R000000001__'+theGuiSample+'__'+NewRelease+'-'+NewLabel+'__'+NewFormat+'.root '
            print "New file on the GUI: "+DqmGuiNewRepository+newGuiFileName
#            os.system('wget --ca-directory $X509_CERT_DIR/ --certificate=$X509_USER_PROXY --private-key=$X509_USER_PROXY '+DqmGuiNewRepository+newGuiFileName)
            os.system('/usr/bin/curl -O -L --capath $X509_CERT_DIR --key $X509_USER_PROXY --cert $X509_USER_PROXY '+DqmGuiNewRepository+newGuiFileName)
            os.system('mv '+newGuiFileName+' '+NewRelease+'/'+NewTag+'/'+sample+'/'+'val.'+sample+'.root')
        elif (GetFilesFrom=='CASTOR'):
            print '*** Getting new file from castor'
            NewCondition=NewCondition+isFastSimNew
            os.system('rfcp '+CastorRepository+'/'+NewRelease+'_'+NewCondition+'_'+sample+'_val.'+sample+'.root '+NewRelease+'/'+NewTag+'/'+sample+'/'+'val.'+sample+'.root')
        elif ((GetFilesFrom=='WEB') & (os.path.isfile(newSampleOnWeb))) :
            print "New file found at: "+newSample+' -> Copy that one'
            os.system('cp '+newSampleOnWeb+' '+NewRelease+'/'+NewTag+'/'+sample)
        else:
            print '*** WARNING: no signal file was found'
        
        if (os.path.isfile(RefRelease+'/'+RefTag+'/'+sample+'/val.'+sample+'.root')==True):
            print "Reference file found at: "+RefRelease+'/'+RefTag+'/'+sample+'/val.'+sample+'.root'+' -> Use that one'
        elif (GetRefsFrom=='GUI'):
            theGuiSample = sample
            # Temporary fix due to the wrong name used for JPsiMM in the DQM GUI
            if ((sample=="RelValJpsiMM")&(RefCondition=='POSTLS1')):
                theGuiSample = "RelValJpsiMMM"
            #
            if (RefCondition=='POSTLS1'):
                theGuiSample = theGuiSample+theGuiPostFixLS1
#            refGuiFileName='DQM_V0001_R000000001__'+sample+'__'+RefRelease+'-'+RefLabel+'__'+RefFormat+'.root '
            refGuiFileName='DQM_V0001_R000000001__'+theGuiSample+'__'+RefRelease+'-'+RefLabel+'__'+RefFormat+'.root '
            print "Ref file on the GUI: "+DqmGuiRefRepository+refGuiFileName
            print '*** Getting reference file from the DQM GUI server'
#            os.system('wget --ca-directory $X509_CERT_DIR/ --certificate=$X509_USER_PROXY --private-key=$X509_USER_PROXY '+DqmGuiRefRepository+refGuiFileName)
            os.system('/usr/bin/curl -O -L --capath $X509_CERT_DIR --key $X509_USER_PROXY --cert $X509_USER_PROXY '+DqmGuiRefRepository+refGuiFileName)
            os.system('mv '+refGuiFileName+' '+RefRelease+'/'+RefTag+'/'+sample+'/'+'val.'+sample+'.root')
        elif (GetRefsFrom=='CASTOR'):
            print '*** Getting reference file from castor'
            RefCondition=RefCondition+isFastSimOld
            print 'rfcp '+CastorRefRepository+'/'+RefRelease+'_'+RefCondition+'_'+sample+'_val.'+sample+'.root '+RefRelease+'/'+RefTag+'/'+sample+'/'+'val.'+sample+'.root'
            os.system('rfcp '+CastorRefRepository+'/'+RefRelease+'_'+RefCondition+'_'+sample+'_val.'+sample+'.root '+RefRelease+'/'+RefTag+'/'+sample+'/'+'val.'+sample+'.root')
        elif ((GetRefsFrom=='WEB') & (os.path.isfile(refSampleOnWeb))):
            print '*** Getting reference file from '+RefRelease
            os.system('cp '+refSampleOnWeb+' '+RefRelease+'/'+RefTag+'/'+sample)
        else:
            print '*** WARNING: no reference file was found'


        cfgFileName=sample+'_'+NewRelease+'_'+RefRelease
        hltcfgFileName='HLT'+sample+'_'+NewRelease+'_'+RefRelease
        seedcfgFileName='DQMSEED'+sample+'_'+NewRelease+'_'+RefRelease
        recocfgFileName='DQMRECO'+sample+'_'+NewRelease+'_'+RefRelease
        recomuoncfgFileName='RECO'+sample+'_'+NewRelease+'_'+RefRelease
        isolcfgFileName='ISOL'+sample+'_'+NewRelease+'_'+RefRelease

        if os.path.isfile(RefRelease+'/'+RefTag+'/'+sample+'/val.'+sample+'.root'):
            replace_map = { 'DATATYPE': 'RECO', 'NEW_FILE':NewRelease+'/'+NewTag+'/'+sample+'/val.'+sample+'.root', 'REF_FILE':RefRelease+'/'+RefTag+'/'+sample+'/val.'+sample+'.root', 'REF_LABEL':sample, 'NEW_LABEL': sample, 'REF_RELEASE':RefRelease, 'NEW_RELEASE':NewRelease, 'REFSELECTION':RefTag, 'NEWSELECTION':NewTag, 'TrackValHistoPublisher': cfgFileName}
            if (ValidateHLT):
                replace_map_HLT = { 'DATATYPE': 'HLT', 'NEW_FILE':NewRelease+'/'+NewTag+'/'+sample+'/val.'+sample+'.root', 'REF_FILE':RefRelease+'/'+RefTag+'/'+sample+'/val.'+sample+'.root', 'REF_LABEL':sample, 'NEW_LABEL': sample, 'REF_RELEASE':RefRelease, 'NEW_RELEASE':NewRelease, 'REFSELECTION':RefTag, 'NEWSELECTION':NewTag, 'TrackValHistoPublisher': hltcfgFileName}
            if (ValidateDQM):
                replace_map_DIST = { 'DATATYPE': 'RECO', 'NEW_FILE':NewRelease+'/'+NewTag+'/'+sample+'/val.'+sample+'.root', 'REF_FILE':RefRelease+'/'+RefTag+'/'+sample+'/val.'+sample+'.root', 'REF_LABEL':sample, 'NEW_LABEL': sample, 'REF_RELEASE':RefRelease, 'NEW_RELEASE':NewRelease, 'REFSELECTION':RefTag, 'NEWSELECTION':NewTag, 'RecoValHistoPublisher': recocfgFileName}
                replace_map_SEED = { 'DATATYPE': 'RECO', 'NEW_FILE':NewRelease+'/'+NewTag+'/'+sample+'/val.'+sample+'.root', 'REF_FILE':RefRelease+'/'+RefTag+'/'+sample+'/val.'+sample+'.root', 'REF_LABEL':sample, 'NEW_LABEL': sample, 'REF_RELEASE':RefRelease, 'NEW_RELEASE':NewRelease, 'REFSELECTION':RefTag, 'NEWSELECTION':NewTag, 'SeedValHistoPublisher': seedcfgFileName}
            if (ValidateISO):
                replace_map_ISOL = { 'DATATYPE': 'RECO', 'NEW_FILE':NewRelease+'/'+NewTag+'/'+sample+'/val.'+sample+'.root', 'REF_FILE':RefRelease+'/'+RefTag+'/'+sample+'/val.'+sample+'.root', 'REF_LABEL':sample, 'NEW_LABEL': sample, 'REF_RELEASE':RefRelease, 'NEW_RELEASE':NewRelease, 'REFSELECTION':RefTag, 'NEWSELECTION':NewTag, 'IsoValHistoPublisher': isolcfgFileName}
            if (ValidateRECO):
                replace_map_RECO = { 'DATATYPE': 'RECO', 'NEW_FILE':NewRelease+'/'+NewTag+'/'+sample+'/val.'+sample+'.root', 'REF_FILE':RefRelease+'/'+RefTag+'/'+sample+'/val.'+sample+'.root', 'IS_FSIM':'', 'REF_LABEL':sample, 'NEW_LABEL': sample, 'REF_RELEASE':RefRelease, 'NEW_RELEASE':NewRelease, 'REFSELECTION':RefTag, 'NEWSELECTION':NewTag, 'RecoMuonValHistoPublisher': recomuoncfgFileName} 
        else:
            print "No reference file found at: ", RefRelease+'/'+RefTag+'/'+sample
            replace_map = { 'DATATYPE': 'RECO', 'NEW_FILE':NewRelease+'/'+NewTag+'/'+sample+'/val.'+sample+'.root', 'REF_FILE':NewRelease+'/'+NewTag+'/'+sample+'/val.'+sample+'.root', 'REF_LABEL':sample, 'NEW_LABEL': sample, 'REF_RELEASE':NewRelease, 'NEW_RELEASE':NewRelease, 'REFSELECTION':NewTag, 'NEWSELECTION':NewTag, 'TrackValHistoPublisher': cfgFileName}
            if (ValidateHLT):
                replace_map_HLT = { 'DATATYPE': 'HLT', 'NEW_FILE':NewRelease+'/'+NewTag+'/'+sample+'/val.'+sample+'.root', 'REF_FILE':NewRelease+'/'+NewTag+'/'+sample+'/val.'+sample+'.root', 'REF_LABEL':sample, 'NEW_LABEL': sample, 'REF_RELEASE':NewRelease, 'NEW_RELEASE':NewRelease, 'REFSELECTION':NewTag, 'NEWSELECTION':NewTag, 'TrackValHistoPublisher': hltcfgFileName}
            if (ValidateDQM):
                replace_map_DIST = { 'DATATYPE': 'RECO', 'NEW_FILE':NewRelease+'/'+NewTag+'/'+sample+'/val.'+sample+'.root', 'REF_FILE':NewRelease+'/'+NewTag+'/'+sample+'/val.'+sample+'.root', 'REF_LABEL':sample, 'NEW_LABEL': sample, 'REF_RELEASE':NewRelease, 'NEW_RELEASE':NewRelease, 'REFSELECTION':NewTag, 'NEWSELECTION':NewTag, 'RecoValHistoPublisher': recocfgFileName}
                replace_map_SEED = { 'DATATYPE': 'RECO', 'NEW_FILE':NewRelease+'/'+NewTag+'/'+sample+'/val.'+sample+'.root', 'REF_FILE':NewRelease+'/'+NewTag+'/'+sample+'/val.'+sample+'.root', 'REF_LABEL':sample, 'NEW_LABEL': sample, 'REF_RELEASE':NewRelease, 'NEW_RELEASE':NewRelease, 'REFSELECTION':NewTag, 'NEWSELECTION':NewTag, 'SeedValHistoPublisher': seedcfgFileName}
            if (ValidateISO):
                replace_map_ISOL = { 'DATATYPE': 'RECO', 'NEW_FILE':NewRelease+'/'+NewTag+'/'+sample+'/val.'+sample+'.root', 'REF_FILE':NewRelease+'/'+NewTag+'/'+sample+'/val.'+sample+'.root', 'REF_LABEL':sample, 'NEW_LABEL': sample, 'REF_RELEASE':NewRelease, 'NEW_RELEASE':NewRelease, 'REFSELECTION':NewTag, 'NEWSELECTION':NewTag, 'IsoValHistoPublisher': isolcfgFileName}
            if (ValidateRECO):
                replace_map_RECO = { 'DATATYPE': 'RECO', 'NEW_FILE':NewRelease+'/'+NewTag+'/'+sample+'/val.'+sample+'.root', 'REF_FILE':NewRelease+'/'+NewTag+'/'+sample+'/val.'+sample+'.root', 'IS_FSIM':'', 'REF_LABEL':sample, 'NEW_LABEL': sample, 'REF_RELEASE':NewRelease, 'NEW_RELEASE':NewRelease, 'REFSELECTION':NewTag, 'NEWSELECTION':NewTag, 'RecoMuonValHistoPublisher': recomuoncfgFileName}

        templatemacroFile = open(macro, 'r')
        macroFile = open(cfgFileName+'.C' , 'w' )
        replace(replace_map, templatemacroFile, macroFile)

        if (ValidateHLT):
            templatemacroFile = open(macro, 'r')
            hltmacroFile = open(hltcfgFileName+'.C' , 'w' )
            replace(replace_map_HLT, templatemacroFile, hltmacroFile)

        if (ValidateDQM):
            templatemacroFile = open(macroReco, 'r')
            recomacroFile = open(recocfgFileName+'.C' , 'w' )
            replace(replace_map_DIST, templatemacroFile, recomacroFile)
            templatemacroFile = open(macroSeed, 'r')
            seedmacroFile = open(seedcfgFileName+'.C' , 'w' )
            replace(replace_map_SEED, templatemacroFile, seedmacroFile)

        if (ValidateISO):
            templatemacroFile = open(macroIsol, 'r')
            isolmacroFile = open(isolcfgFileName+'.C' , 'w' )
            replace(replace_map_ISOL, templatemacroFile, isolmacroFile)

        if (ValidateRECO):
            templatemacroFile = open(macroMuonReco, 'r')
            recomuonmacroFile = open(recomuoncfgFileName+'.C' , 'w' )
            replace(replace_map_RECO, templatemacroFile, recomuonmacroFile)

        if(Submit):
            os.system('root -b -q -l '+ cfgFileName+'.C'+ '>  macro.'+cfgFileName+'.log')
            if (ValidateHLT):
                os.system('root -b -q -l '+ hltcfgFileName+'.C'+ '>  macro.'+hltcfgFileName+'.log')
            if (ValidateDQM):
                os.system('root -b -q -l '+ recocfgFileName+'.C'+ '>  macro.'+recocfgFileName+'.log')
                os.system('root -b -q -l '+ seedcfgFileName+'.C'+ '>  macro.'+seedcfgFileName+'.log')
            if (ValidateISO):
                os.system('root -b -q -l '+ isolcfgFileName+'.C'+ '>  macro.'+isolcfgFileName+'.log')
                if (NewFastSim&RefFastSim):
                    os.system('mv '+NewRelease+'/'+NewTag+'/'+sample+'/MuonIsolationV_inc.pdf '+NewRelease+'/'+NewTag+'/'+sample+'/MuonIsolationV_inc_FS.pdf')
            if (ValidateRECO):
                os.system('root -b -q -l '+ recomuoncfgFileName+'.C'+ '>  macro.'+recomuoncfgFileName+'.log')
                if (NewFastSim&RefFastSim):
                    os.system('mv '+NewRelease+'/'+NewTag+'/'+sample+'/RecoMuonV.pdf '+NewRelease+'/'+NewTag+'/'+sample+'/RecoMuonV_FS.pdf')

        if(Publish):
            newdir=WebRepository+'/'+NewRelease+'/'+NewTag+'/'+sample 
            if(os.path.exists(newdir)==False):
                os.makedirs(newdir)
            os.system('rm '+NewRelease+'/'+NewTag+'/'+sample+'/val.'+sample+'.root')  
            os.system('scp -r '+NewRelease+'/'+NewTag+'/'+sample+'/* ' + newdir)
