#! /usr/bin/env python

import os
import sys
import fileinput
import string

NewVersion='4_2_0_pre5'
RefVersion='4_2_0_pre4'
NewRelease='CMSSW_'+NewVersion
RefRelease='CMSSW_'+RefVersion
#NewRelease='Summer09'
#RefRelease='Summer09_pre1'

NewCondition='MC'
RefCondition='MC'
#NewCondition='STARTUP'
#RefCondition='STARTUP'

NewFastSim=False
RefFastSim=False

if (NewCondition=='MC'):
    samples= ['RelValSingleMuPt10','RelValSingleMuPt100','RelValSingleMuPt1000','RelValTTbar']
    if (NewFastSim|RefFastSim):
        samples= ['RelValSingleMuPt10','RelValSingleMuPt100','RelValTTbar']
elif (NewCondition=='STARTUP'):
    samples= ['RelValTTbar','RelValZMM','RelValJpsiMM']
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

DqmGuiNewRepository = 'https://cmsweb.cern.ch/dqm/dev/data/browse/Development/RelVal/CMSSW_4_2_x/'
#DqmGuiRefRepository = 'https://cmsweb.cern.ch/dqm/offline/data/browse/ROOT/RelVal/CMSSW_4_2_x/'
DqmGuiRefRepository = 'https://cmsweb.cern.ch/dqm/dev/data/browse/Development/RelVal/CMSSW_4_2_x/'
CastorRepository = '/castor/cern.ch/user/a/aperrott/ValidationRecoMuon'
if ((GetFilesFrom=='GUI')|(GetRefsFrom=='GUI')):
    print "*** Did you remind doing:"

# USE THIS WITH wget
#    print " > source /afs/cern.ch/cms/LCG/LCG-2/UI/cms_ui_env.(c)sh"
# USE THIS WITH curl
    print " > source /afs/cern.ch/project/gd/LCG-share/sl5/etc/profile.d/grid_env.(c)sh"
    print " > voms-proxy-init"


# These are only needed if you copy any root file from the DQM GUI:
NewLabel='MC_42_V3'
if (NewCondition=='STARTUP'):
    NewLabel='START42_V3'
RefLabel='MC_42_V1'
if (RefCondition=='STARTUP'):
    RefLabel='START42_V1'


ValidateHLT=True
if (NewFastSim|RefFastSim):
    ValidateDQM=False
    ValidateISO=True
else:
    ValidateDQM=True
    ValidateISO=True



NewFormat='GEN-SIM-RECO'
RefFormat='GEN-SIM-RECO'
NewTag = NewCondition+'_noPU_ootb'
RefTag = RefCondition+'_noPU_ootb'
if (NewFastSim):
    NewTag = NewTag+'_FSIM'
    NewCondition=NewCondition+'_FSIM'
    NewLabel=NewLabel+'_FastSim'
    NewFormat='GEN-SIM-DIGI-RECO'
if (RefFastSim):
    RefTag = RefTag+'_FSIM'
    RefCondition=RefCondition+'_FSIM'
    RefLabel=RefLabel+'_FastSim'
    RefFormat='GEN-SIM-DIGI-RECO'

NewLabel=NewLabel+'-v1'
RefLabel=RefLabel+'-v1'


WebRepository = '/afs/cern.ch/cms/Physics/muon/CMSSW/Performance/RecoMuon/Validation/val'
CastorRefRepository = '/castor/cern.ch/user/a/aperrott/ValidationRecoMuon'    

macro='macro/TrackValHistoPublisher.C'
macroSeed='macro/SeedValHistoPublisher.C'
macroReco='macro/RecoValHistoPublisher.C'
macroIsol='macro/IsoValHistoPublisher.C'

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
#            os.system('wget --ca-directory $X509_CERT_DIR/ --certificate=$X509_USER_PROXY --private-key=$X509_USER_PROXY '+DqmGuiNewRepository+'DQM_V0001_R000000001__'+sample+'__'+NewRelease+'-'+NewLabel+'__'+NewFormat+'.root ')
            os.system('/usr/bin/curl -O -L --capath $X509_CERT_DIR --key $X509_USER_PROXY --cert $X509_USER_PROXY '+DqmGuiNewRepository+'DQM_V0001_R000000001__'+sample+'__'+NewRelease+'-'+NewLabel+'__'+NewFormat+'.root ')
            os.system('mv DQM_V0001_R000000001__'+sample+'__'+NewRelease+'-'+NewLabel+'__'+NewFormat+'.root '+NewRelease+'/'+NewTag+'/'+sample+'/'+'val.'+sample+'.root')
        elif (GetFilesFrom=='CASTOR'):
            os.system('rfcp '+CastorRepository+'/'+NewRelease+'_'+NewCondition+'_'+sample+'_val.'+sample+'.root '+NewRelease+'/'+NewTag+'/'+sample+'/'+'val.'+sample+'.root')
        elif ((GetFilesFrom=='WEB') & (os.path.isfile(newSampleOnWeb))) :
            print "New file found at: "+newSample+' -> Copy that one'
            os.system('cp '+newSampleOnWeb+' '+NewRelease+'/'+NewTag+'/'+sample)
        else:
            print '*** WARNING: no signal file was found'

        
        if (os.path.isfile(RefRelease+'/'+RefTag+'/'+sample+'/val.'+sample+'.root')==True):
            print "Reference file found at: "+RefRelease+'/'+RefTag+'/'+sample+'/val.'+sample+'.root'+' -> Use that one'
        elif (GetRefsFrom=='GUI'):
            print '*** Getting reference file from the DQM GUI server'
#            os.system('wget --ca-directory $X509_CERT_DIR/ --certificate=$X509_USER_PROXY --private-key=$X509_USER_PROXY '+DqmGuiRefRepository+'DQM_V0001_R000000001__'+sample+'__'+RefRelease+'-'+RefLabel+'__'+RefFormat+'.root ')
            os.system('/usr/bin/curl -O -L --capath $X509_CERT_DIR --key $X509_USER_PROXY --cert $X509_USER_PROXY '+DqmGuiRefRepository+'DQM_V0001_R000000001__'+sample+'__'+RefRelease+'-'+RefLabel+'__'+RefFormat+'.root ')
            os.system('mv DQM_V0001_R000000001__'+sample+'__'+RefRelease+'-'+RefLabel+'__'+RefFormat+'.root '+RefRelease+'/'+RefTag+'/'+sample+'/'+'val.'+sample+'.root')
        elif (GetRefsFrom=='CASTOR'):
            print '*** Getting reference file from castor'
            os.system('rfcp '+CastorRefRepository+'/'+RefRelease+'_'+RefCondition+'_'+sample+'_val.'+sample+'.root '+RefRelease+'/'+RefTag+'/'+sample+'/'+'val.'+sample+'.root')
        elif ((GetRefsFrom=='WEB') & (os.path.isfile(refSampleOnWeb))):
            print '*** Getting reference file from '+RefRelease
            os.system('cp '+refSampleOnWeb+' '+RefRelease+'/'+RefTag+'/'+sample)
        else:
            print '*** WARNING: no reference file was found'


        cfgFileName=sample+'_'+NewRelease+'_'+RefRelease
        hltcfgFileName='HLT'+sample+'_'+NewRelease+'_'+RefRelease
        seedcfgFileName='SEED'+sample+'_'+NewRelease+'_'+RefRelease
        recocfgFileName='RECO'+sample+'_'+NewRelease+'_'+RefRelease
        isolcfgFileName='ISOL'+sample+'_'+NewRelease+'_'+RefRelease

        if os.path.isfile(RefRelease+'/'+RefTag+'/'+sample+'/val.'+sample+'.root'):
            replace_map_RECO = { 'DATATYPE': 'RECO', 'NEW_FILE':NewRelease+'/'+NewTag+'/'+sample+'/val.'+sample+'.root', 'REF_FILE':RefRelease+'/'+RefTag+'/'+sample+'/val.'+sample+'.root', 'REF_LABEL':sample, 'NEW_LABEL': sample, 'REF_RELEASE':RefRelease, 'NEW_RELEASE':NewRelease, 'REFSELECTION':RefTag, 'NEWSELECTION':NewTag, 'TrackValHistoPublisher': cfgFileName}
            if (ValidateHLT):
                replace_map_HLT = { 'DATATYPE': 'HLT', 'NEW_FILE':NewRelease+'/'+NewTag+'/'+sample+'/val.'+sample+'.root', 'REF_FILE':RefRelease+'/'+RefTag+'/'+sample+'/val.'+sample+'.root', 'REF_LABEL':sample, 'NEW_LABEL': sample, 'REF_RELEASE':RefRelease, 'NEW_RELEASE':NewRelease, 'REFSELECTION':RefTag, 'NEWSELECTION':NewTag, 'TrackValHistoPublisher': hltcfgFileName}
            if (ValidateDQM):
                replace_map_DIST = { 'DATATYPE': 'RECO', 'NEW_FILE':NewRelease+'/'+NewTag+'/'+sample+'/val.'+sample+'.root', 'REF_FILE':RefRelease+'/'+RefTag+'/'+sample+'/val.'+sample+'.root', 'REF_LABEL':sample, 'NEW_LABEL': sample, 'REF_RELEASE':RefRelease, 'NEW_RELEASE':NewRelease, 'REFSELECTION':RefTag, 'NEWSELECTION':NewTag, 'RecoValHistoPublisher': recocfgFileName}
                replace_map_SEED = { 'DATATYPE': 'RECO', 'NEW_FILE':NewRelease+'/'+NewTag+'/'+sample+'/val.'+sample+'.root', 'REF_FILE':RefRelease+'/'+RefTag+'/'+sample+'/val.'+sample+'.root', 'REF_LABEL':sample, 'NEW_LABEL': sample, 'REF_RELEASE':RefRelease, 'NEW_RELEASE':NewRelease, 'REFSELECTION':RefTag, 'NEWSELECTION':NewTag, 'SeedValHistoPublisher': seedcfgFileName}
            if (ValidateISO):
                replace_map_ISOL = { 'DATATYPE': 'RECO', 'NEW_FILE':NewRelease+'/'+NewTag+'/'+sample+'/val.'+sample+'.root', 'REF_FILE':RefRelease+'/'+RefTag+'/'+sample+'/val.'+sample+'.root', 'REF_LABEL':sample, 'NEW_LABEL': sample, 'REF_RELEASE':RefRelease, 'NEW_RELEASE':NewRelease, 'REFSELECTION':RefTag, 'NEWSELECTION':NewTag, 'IsoValHistoPublisher': isolcfgFileName}
        else:
            print "No reference file found at: ", RefRelease+'/'+RefTag+'/'+sample
            replace_map_RECO = { 'DATATYPE': 'RECO', 'NEW_FILE':NewRelease+'/'+NewTag+'/'+sample+'/val.'+sample+'.root', 'REF_FILE':NewRelease+'/'+NewTag+'/'+sample+'/val.'+sample+'.root', 'REF_LABEL':sample, 'NEW_LABEL': sample, 'REF_RELEASE':NewRelease, 'NEW_RELEASE':NewRelease, 'REFSELECTION':NewTag, 'NEWSELECTION':NewTag, 'TrackValHistoPublisher': cfgFileName}
            if (ValidateHLT):
                replace_map_HLT = { 'DATATYPE': 'HLT', 'NEW_FILE':NewRelease+'/'+NewTag+'/'+sample+'/val.'+sample+'.root', 'REF_FILE':NewRelease+'/'+NewTag+'/'+sample+'/val.'+sample+'.root', 'REF_LABEL':sample, 'NEW_LABEL': sample, 'REF_RELEASE':NewRelease, 'NEW_RELEASE':NewRelease, 'REFSELECTION':NewTag, 'NEWSELECTION':NewTag, 'TrackValHistoPublisher': hltcfgFileName}
            if (ValidateDQM):
                replace_map_DIST = { 'DATATYPE': 'RECO', 'NEW_FILE':NewRelease+'/'+NewTag+'/'+sample+'/val.'+sample+'.root', 'REF_FILE':NewRelease+'/'+NewTag+'/'+sample+'/val.'+sample+'.root', 'REF_LABEL':sample, 'NEW_LABEL': sample, 'REF_RELEASE':NewRelease, 'NEW_RELEASE':NewRelease, 'REFSELECTION':NewTag, 'NEWSELECTION':NewTag, 'RecoValHistoPublisher': recocfgFileName}
                replace_map_SEED = { 'DATATYPE': 'RECO', 'NEW_FILE':NewRelease+'/'+NewTag+'/'+sample+'/val.'+sample+'.root', 'REF_FILE':NewRelease+'/'+NewTag+'/'+sample+'/val.'+sample+'.root', 'REF_LABEL':sample, 'NEW_LABEL': sample, 'REF_RELEASE':NewRelease, 'NEW_RELEASE':NewRelease, 'REFSELECTION':NewTag, 'NEWSELECTION':NewTag, 'SeedValHistoPublisher': seedcfgFileName}
            if (ValidateISO):
                replace_map_ISOL = { 'DATATYPE': 'RECO', 'NEW_FILE':NewRelease+'/'+NewTag+'/'+sample+'/val.'+sample+'.root', 'REF_FILE':NewRelease+'/'+NewTag+'/'+sample+'/val.'+sample+'.root', 'REF_LABEL':sample, 'NEW_LABEL': sample, 'REF_RELEASE':NewRelease, 'NEW_RELEASE':NewRelease, 'REFSELECTION':NewTag, 'NEWSELECTION':NewTag, 'IsoValHistoPublisher': isolcfgFileName}

        templatemacroFile = open(macro, 'r')
        macroFile = open(cfgFileName+'.C' , 'w' )
        replace(replace_map_RECO, templatemacroFile, macroFile)

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

        if(Submit):
            os.system('root -b -q -l '+ cfgFileName+'.C'+ '>  macro.'+cfgFileName+'.log')
            if (ValidateHLT):
                os.system('root -b -q -l '+ hltcfgFileName+'.C'+ '>  macro.'+hltcfgFileName+'.log')
            if (ValidateDQM):
                os.system('root -b -q -l '+ recocfgFileName+'.C'+ '>  macro.'+recocfgFileName+'.log')
                os.system('root -b -q -l '+ seedcfgFileName+'.C'+ '>  macro.'+seedcfgFileName+'.log')
            if (ValidateISO):
                os.system('root -b -q -l '+ isolcfgFileName+'.C'+ '>  macro.'+isolcfgFileName+'.log')

        if(Publish):
            newdir=WebRepository+'/'+NewRelease+'/'+NewTag+'/'+sample 
            if(os.path.exists(newdir)==False):
                os.makedirs(newdir)
            os.system('rm '+NewRelease+'/'+NewTag+'/'+sample+'/val.'+sample+'.root')  
            os.system('scp -r '+NewRelease+'/'+NewTag+'/'+sample+'/* ' + newdir)
