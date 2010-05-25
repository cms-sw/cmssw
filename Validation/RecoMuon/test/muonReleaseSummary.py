#! /usr/bin/env python

import os
import sys
import fileinput
import string

NewVersion='3_6_1'
RefVersion='3_6_0'
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
# These are some of the (pre)production samples, to be included by hand:
#samples= ['ppMuXLoose', 'InclusiveMu5_Pt50', 'InclusiveMu5_Pt250', 'ZmumuJet_Pt0to15', 'ZmumuJet_Pt300toInf', 'ZmumuJet_Pt80to120']
#samples= ['InclusiveMu5_Pt50', 'ZmumuJet_Pt0to15', 'ZmumuJet_Pt300toInf', 'ZmumuJet_Pt80to120']

Submit=True
Publish=False

GetFilesFromCastor=True
GetRefsFromCastor=True
#CastorRepository = '/castor/cern.ch/cms/store/temp/dqm/offline/harvesting_output/mc/relval'
CastorRepository = '/castor/cern.ch/user/a/aperrott/ValidationRecoMuon'
### Older repositories:
#CastorRepository = '/castor/cern.ch/user/n/nuno/relval/harvest'
#CastorRepository = '/castor/cern.ch/user/n/nuno/preproduction/harvest'
#CastorRepository = '/castor/cern.ch/user/j/jhegeman/preproduction_summer09/3_1_2'

ValidateHLT=True
if (NewFastSim|RefFastSim):
    ValidateDQM=False
    ValidateISO=True
else:
    ValidateDQM=True
    ValidateISO=True

if (NewFastSim):
    NewTag = NewCondition+'_noPU_ootb_FSIM'
    NewLabel=NewCondition+'MC_36Y_V2_FastSim'
    if (NewCondition=='STARTUP'):
        NewLabel=NewCondition+'START36_V2_FastSim'
    NewFormat='GEN-SIM-DIGI-RECO'
else:
    NewTag = NewCondition+'_noPU_ootb'
    NewLabel=NewCondition+'MC_36Y_V2'
    if (NewCondition=='STARTUP'):
        NewLabel=NewCondition+'START36_V2'
    NewFormat='GEN-SIM-RECO'

if (RefFastSim):
    RefTag = RefCondition+'_noPU_ootb_FSIM'
else:
    RefTag = RefCondition+'_noPU_ootb'

NewLabel=NewLabel+'-v1'

if (NewFastSim):
    NewCondition=NewCondition+'_FSIM'
if (RefFastSim):
    RefCondition=RefCondition+'_FSIM'


NewRepository = '/afs/cern.ch/cms/Physics/muon/CMSSW/Performance/RecoMuon/Validation/val'
RefRepository = '/afs/cern.ch/cms/Physics/muon/CMSSW/Performance/RecoMuon/Validation/val'
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

    newdir=NewRepository+'/'+NewRelease+'/'+NewTag+'/'+sample 


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
        newSample=NewRepository+'/'+NewRelease+'/'+NewTag+'/'+sample+'/'+'val.'+sample+'.root'
        refSample=RefRepository+'/'+RefRelease+'/'+RefTag+'/'+sample+'/'+'val.'+sample+'.root'

        if (os.path.isfile(NewRelease+'/'+NewTag+'/'+sample+'/val'+sample+'.root')==True):
            # FOR SOME REASON THIS DOES NOT WORK: to be checked...
            print "New file found at: ",NewRelease+'/'+NewTag+'/'+sample+'/val'+sample+'.root'+' -> Use that one'

        elif (GetFilesFromCastor):
# Check the number of events in the harvested samples, needed to retrieve the path on castor
            if (CastorRepository=='/castor/cern.ch/user/a/aperrott/ValidationRecoMuon'):
                os.system('rfcp '+CastorRepository+'/'+NewRelease+'_'+NewCondition+'_'+sample+'_val.'+sample+'.root '+NewRelease+'/'+NewTag+'/'+sample+'/'+'val.'+sample+'.root')
            else:
                if (NewFastSim):
                    NEVT='27000'
                else:
                    NEVT='9000'
                    if (sample=='RelValSingleMuPt10'):
                        NEVT='25000'
                    elif(sample=='RelValZMM'):
                        NEVT='8995'
                    elif((sample=='RelValTTbar')&(NewCondition=='STARTUP')):
                        NEVT='34000'
                    os.system('rfcp '+CastorRepository+'/'+NewVersion+'/'+sample+'__'+NewRelease+'-'+NewLabel+'__'+NewFormat+'/run_1/nevents_'+NEVT+'/DQM_V0001_R000000001__'+sample+'__'+NewRelease+'-'+NewLabel+'__'+NewFormat+'_1.root '+NewRelease+'/'+NewTag+'/'+sample+'/'+'val.'+sample+'.root')

        elif (os.path.isfile(newSample)) :
            os.system('cp '+newSample+' '+NewRelease+'/'+NewTag+'/'+sample)

             
        if (os.path.isfile(RefRelease+'/'+RefTag+'/'+sample+'/val'+sample+'.root')!=True and os.path.isfile(refSample)):
            print '*** Getting reference file from '+RefRelease
            os.system('cp '+refSample+' '+RefRelease+'/'+RefTag+'/'+sample)
        elif (GetRefsFromCastor):
            print '*** Getting reference file from castor'
            os.system('rfcp '+CastorRefRepository+'/'+RefRelease+'_'+RefCondition+'_'+sample+'_val.'+sample+'.root '+RefRelease+'/'+RefTag+'/'+sample+'/'+'val.'+sample+'.root')
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
            if(os.path.exists(newdir)==False):
                os.makedirs(newdir)
            os.system('rm '+NewRelease+'/'+NewTag+'/'+sample+'/val.'+sample+'.root')  
            os.system('scp -r '+NewRelease+'/'+NewTag+'/'+sample+'/* ' + newdir)
