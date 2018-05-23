#! /usr/bin/env python

import os
import sys
import fileinput
import string

NewVersion='3_6_1_patch3_seedFix'
RefVersion='3_6_1_patch3_orig'
#NewVersion='3_6_1_patch3-seedFix'
#RefVersion='3_6_1_patch3-orig'
NewRelease='CMSSW_'+NewVersion
RefRelease='CMSSW_'+RefVersion
#NewRelease='Summer09'
#RefRelease='Summer09_pre1'

NewFastSim=False
RefFastSim=False

samples= ['RelValJpsiMM_mc',
          'RelValJpsiMM_startup',
          'RelValQCD_FlatPt_15_3000_mc',
          'RelValQCD_FlatPt_15_3000_startup',
          'RelValSingleMuPt1000_mc',
          'RelValSingleMuPt1000_startup',
          'RelValSingleMuPt100_mc',
          'RelValSingleMuPt100_startup',
          'RelValSingleMuPt10_mc',
          'RelValSingleMuPt10_startup',
          'RelValZMM_mc',
          'RelValZMM_startup'
          ]

#samples= ['InclusiveMu5_Pt50', 'ZmumuJet_Pt0to15', 'ZmumuJet_Pt300toInf', 'ZmumuJet_Pt80to120']

Submit=True
Publish=False

GetFilesFromCastor=True
GetRefsFromCastor=True
CastorRepository = '/castor/cern.ch/user/s/slava77/reltest/CMSSW_3_6_1_patch3-seedFix/harvesting'
#CastorRepository = '/castor/cern.ch/user/a/aperrott/ValidationRecoMuon'

ValidateHLT=True
ValidateDQM=True
ValidateISO=True

NewTag = ''
RefTag = ''


#CastorRefRepository = '/castor/cern.ch/user/a/aperrott/ValidationRecoMuon'
CastorRefRepository = '/castor/cern.ch/user/s/slava77/reltest/CMSSW_3_6_1_patch3-orig/harvesting'


macro='macro/TrackValHistoPublisher.C'
macroSeed='macro/SeedValHistoPublisher.C'
macroReco='macro/RecoValHistoPublisher.C'
macroIsol='macro/IsoValHistoPublisher.C'

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

###############################
NewRepository = '/afs/cern.ch/cms/Physics/muon/CMSSW/Performance/RecoMuon/Validation/val'
RefRepository = '/afs/cern.ch/cms/Physics/muon/CMSSW/Performance/RecoMuon/Validation/val'

for sample in samples :

    newdir=NewRepository+'/'+NewRelease+'/'+sample 


    if(os.path.exists(NewRelease+'/'+sample)==False):
        os.makedirs(NewRelease+'/'+sample)

    if(os.path.exists(RefRelease+'/'+sample)==False):
        os.makedirs(RefRelease+'/'+sample)

    checkFile = NewRelease+'/'+sample+'/general_tpToTkmuAssociation.pdf'
    if (RefFastSim):
        checkFile = NewRelease+'/'+sample+'/general_tpToTkmuAssociationFS.pdf'
    if (os.path.isfile(checkFile)==True):
        print "Files of type "+checkFile+' exist alredy: delete them first, if you really want to overwrite them'
    else:
        newSample=NewRepository+'/'+NewRelease+'/'+sample+'/'+'val.'+sample+'.root'
        refSample=RefRepository+'/'+RefRelease+'/'+sample+'/'+'val.'+sample+'.root'

        if (os.path.isfile(NewRelease+'/'+sample+'/val'+sample+'.root')==True):
            # FOR SOME REASON THIS DOES NOT WORK: to be checked...
            print "New file found at: ",NewRelease+'/'+sample+'/val'+sample+'.root'+' -> Use that one'

        elif (GetFilesFromCastor):
            print '*** Getting file from '+NewRelease
            os.system('rfcp '+CastorRepository+'/DQM_CMSSW_3_6_1_patch3-seedFix_'+sample+'.list.root '+NewRelease+'/'+sample+'/'+'val.'+sample+'.root')
        elif (os.path.isfile(newSample)) :
            os.system('cp '+newSample+' '+NewRelease+'/'+sample)

             
        if (os.path.isfile(RefRelease+'/'+sample+'/val'+sample+'.root')!=True and os.path.isfile(refSample)):
            print '*** Getting reference file from '+RefRelease
            os.system('cp '+refSample+' '+RefRelease+'/'+sample)
        elif (GetRefsFromCastor):
            print '*** Getting reference file from castor'
            os.system('rfcp '+CastorRefRepository+'/DQM_CMSSW_3_6_1_patch3-orig_'+sample+'.list.root '+RefRelease+'/'+sample+'/'+'val.'+sample+'.root')
        else:
            print '*** WARNING: no reference file was found'

        cfgFileName=sample+'_'+NewRelease+'_'+RefRelease
        hltcfgFileName='HLT'+sample+'_'+NewRelease+'_'+RefRelease
        seedcfgFileName='SEED'+sample+'_'+NewRelease+'_'+RefRelease
        recocfgFileName='RECO'+sample+'_'+NewRelease+'_'+RefRelease
        isolcfgFileName='ISOL'+sample+'_'+NewRelease+'_'+RefRelease

        if os.path.isfile(RefRelease+'/'+sample+'/val.'+sample+'.root'):
            replace_map_RECO = { 'DATATYPE': 'RECO', 'NEW_FILE':NewRelease+'/'+sample+'/val.'+sample+'.root', 'REF_FILE':RefRelease+'/'+sample+'/val.'+sample+'.root', 'REF_LABEL':sample, 'NEW_LABEL': sample, 'REF_RELEASE':RefRelease, 'NEW_RELEASE':NewRelease, 'REFSELECTION':RefTag, 'NEWSELECTION':NewTag, 'TrackValHistoPublisher': cfgFileName}
            if (ValidateHLT):
                replace_map_HLT = { 'DATATYPE': 'HLT', 'NEW_FILE':NewRelease+'/'+sample+'/val.'+sample+'.root', 'REF_FILE':RefRelease+'/'+sample+'/val.'+sample+'.root', 'REF_LABEL':sample, 'NEW_LABEL': sample, 'REF_RELEASE':RefRelease, 'NEW_RELEASE':NewRelease, 'REFSELECTION':RefTag, 'NEWSELECTION':NewTag, 'TrackValHistoPublisher': hltcfgFileName}
            if (ValidateDQM):
                replace_map_DIST = { 'DATATYPE': 'RECO', 'NEW_FILE':NewRelease+'/'+sample+'/val.'+sample+'.root', 'REF_FILE':RefRelease+'/'+sample+'/val.'+sample+'.root', 'REF_LABEL':sample, 'NEW_LABEL': sample, 'REF_RELEASE':RefRelease, 'NEW_RELEASE':NewRelease, 'REFSELECTION':RefTag, 'NEWSELECTION':NewTag, 'RecoValHistoPublisher': recocfgFileName}
                replace_map_SEED = { 'DATATYPE': 'RECO', 'NEW_FILE':NewRelease+'/'+sample+'/val.'+sample+'.root', 'REF_FILE':RefRelease+'/'+sample+'/val.'+sample+'.root', 'REF_LABEL':sample, 'NEW_LABEL': sample, 'REF_RELEASE':RefRelease, 'NEW_RELEASE':NewRelease, 'REFSELECTION':RefTag, 'NEWSELECTION':NewTag, 'SeedValHistoPublisher': seedcfgFileName}
            if (ValidateISO):
                replace_map_ISOL = { 'DATATYPE': 'RECO', 'NEW_FILE':NewRelease+'/'+sample+'/val.'+sample+'.root', 'REF_FILE':RefRelease+'/'+sample+'/val.'+sample+'.root', 'REF_LABEL':sample, 'NEW_LABEL': sample, 'REF_RELEASE':RefRelease, 'NEW_RELEASE':NewRelease, 'REFSELECTION':RefTag, 'NEWSELECTION':NewTag, 'IsoValHistoPublisher': isolcfgFileName}
        else:
            print "No reference file found at: ", RefRelease+'/'+sample
            replace_map_RECO = { 'DATATYPE': 'RECO', 'NEW_FILE':NewRelease+'/'+sample+'/val.'+sample+'.root', 'REF_FILE':NewRelease+'/'+sample+'/val.'+sample+'.root', 'REF_LABEL':sample, 'NEW_LABEL': sample, 'REF_RELEASE':NewRelease, 'NEW_RELEASE':NewRelease, 'REFSELECTION':NewTag, 'NEWSELECTION':NewTag, 'TrackValHistoPublisher': cfgFileName}
            if (ValidateHLT):
                replace_map_HLT = { 'DATATYPE': 'HLT', 'NEW_FILE':NewRelease+'/'+sample+'/val.'+sample+'.root', 'REF_FILE':NewRelease+'/'+sample+'/val.'+sample+'.root', 'REF_LABEL':sample, 'NEW_LABEL': sample, 'REF_RELEASE':NewRelease, 'NEW_RELEASE':NewRelease, 'REFSELECTION':NewTag, 'NEWSELECTION':NewTag, 'TrackValHistoPublisher': hltcfgFileName}
            if (ValidateDQM):
                replace_map_DIST = { 'DATATYPE': 'RECO', 'NEW_FILE':NewRelease+'/'+sample+'/val.'+sample+'.root', 'REF_FILE':NewRelease+'/'+sample+'/val.'+sample+'.root', 'REF_LABEL':sample, 'NEW_LABEL': sample, 'REF_RELEASE':NewRelease, 'NEW_RELEASE':NewRelease, 'REFSELECTION':NewTag, 'NEWSELECTION':NewTag, 'RecoValHistoPublisher': recocfgFileName}
                replace_map_SEED = { 'DATATYPE': 'RECO', 'NEW_FILE':NewRelease+'/'+sample+'/val.'+sample+'.root', 'REF_FILE':NewRelease+'/'+sample+'/val.'+sample+'.root', 'REF_LABEL':sample, 'NEW_LABEL': sample, 'REF_RELEASE':NewRelease, 'NEW_RELEASE':NewRelease, 'REFSELECTION':NewTag, 'NEWSELECTION':NewTag, 'SeedValHistoPublisher': seedcfgFileName}
            if (ValidateISO):
                replace_map_ISOL = { 'DATATYPE': 'RECO', 'NEW_FILE':NewRelease+'/'+sample+'/val.'+sample+'.root', 'REF_FILE':NewRelease+'/'+sample+'/val.'+sample+'.root', 'REF_LABEL':sample, 'NEW_LABEL': sample, 'REF_RELEASE':NewRelease, 'NEW_RELEASE':NewRelease, 'REFSELECTION':NewTag, 'NEWSELECTION':NewTag, 'IsoValHistoPublisher': isolcfgFileName}

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
            os.system('rm '+NewRelease+'/'+sample+'/val.'+sample+'.root')  
            os.system('scp -r '+NewRelease+'/'+sample+'/* ' + newdir)
