#! /usr/bin/env python

import os
import sys
import fileinput
import string

#NewRelease='Summer09'
#RefRelease='Summer09_pre1'
NewRelease='CMSSW_3_2_3'
RefRelease='CMSSW_3_2_2'

samples= ['RelValSingleMuPt10','RelValSingleMuPt100','RelValSingleMuPt1000','RelValTTbar']
#samples= ['RelValTTbar','RelValZMM']
#samples= ['RelValCosmics']

#samples= ['ppMuXLoose', 'InclusiveMu5_Pt50', 'InclusiveMu5_Pt250', 'ZmumuJet_Pt0to15', 'ZmumuJet_Pt300toInf', 'ZmumuJet_Pt80to120']
#samples= ['InclusiveMu5_Pt50', 'ZmumuJet_Pt0to15', 'ZmumuJet_Pt300toInf', 'ZmumuJet_Pt80to120']

Submit=True
Publish=False

NewFastSim=False
RefFastSim=False

GetFilesFromCastor=False
CastorRepository = '/castor/cern.ch/user/n/nuno/relval/harvest'
#CastorRepository = '/castor/cern.ch/user/n/nuno/preproduction/harvest'
#CastorRepository = '/castor/cern.ch/user/j/jhegeman/preproduction_summer09/3_1_2'

ValidateHLT=True

NewCondition='MC'
RefCondition='MC'
#NewCondition='STARTUP'
#RefCondition='STARTUP'

if (NewFastSim):
    NewTag = NewCondition+'_noPU_ootb_FSIM'
    NewLabel=NewCondition+'_31X_FastSim_v1'
    NewFormat='GEN-SIM-DIGI-RECO'
else:
    NewTag = NewCondition+'_noPU_ootb'
    NewLabel=NewCondition+'_31X_V3-v1'
#    NewLabel=NewCondition+'31X_V3_preproduction_312-v1'
    if (NewCondition=='STARTUP'):
        NewLabel=NewCondition+'31X_V2-v1'
    NewFormat='GEN-SIM-RECO'

if (RefFastSim):
    RefTag = RefCondition+'_noPU_ootb_FSIM'
    RefLabel=RefCondition+'_31X_FastSim_v1'
    RefFormat='GEN-SIM-DIGI-RECO'
else:
    RefTag = RefCondition+'_noPU_ootb'
    RefLabel=RefCondition+'_31X_V3-v1'
#    RefLabel=RefCondition+'_31X_V2_preproduction_311-v1'
    if (RefCondition=='STARTUP'):
        RefLabel=RefCondition+'31X_V2-v1'
    RefFormat='GEN-SIM-RECO'


RefRepository = '/afs/cern.ch/cms/Physics/muon/CMSSW/Performance/RecoMuon/Validation/val'
NewRepository = '/afs/cern.ch/cms/Physics/muon/CMSSW/Performance/RecoMuon/Validation/val'


macro='macro/TrackValHistoPublisher.C'

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

     checkFile = NewRelease+'/'+NewTag+'/'+sample+'/globalMuons_tpToGlbAssociation.pdf'
     if (NewFastSim):
         checkFile = NewRelease+'/'+NewTag+'/'+sample+'/globalMuons_tpToGlbAssociationFS.pdf'
     if (os.path.isfile(checkFile)==True):
         print "Files of type "+checkFile+' exist alredy: delete them first, if you really want to overwrite them'
     else:
         newSample=NewRepository+'/'+NewRelease+'/'+NewTag+'/'+sample+'/'+'val.'+sample+'.root'
         refSample=RefRepository+'/'+RefRelease+'/'+RefTag+'/'+sample+'/'+'val.'+sample+'.root'

         if (os.path.isfile(NewRelease+'/'+NewTag+'/'+sample+'/val'+sample+'.root')==True):
             # FOR SOME REASON THIS DOES NOT WORK: to be checked...
             print "New file found at: ",NewRelease+'/'+NewTag+'/'+sample+'/val'+sample+'.root'+' -> Use that one'
         elif (GetFilesFromCastor):
             os.system('rfcp '+CastorRepository+'/'+NewRelease+'/DQM_V0001_R000000001__'+sample+'__'+NewRelease+'-'+NewLabel+'__'+NewFormat+'.root '+NewRelease+'/'+NewTag+'/'+sample+'/'+'val.'+sample+'.root')
#preprod-hegeman              os.system('rfcp '+CastorRepository+'/DQM_V0001_R000000001__'+sample+'__'+NewRelease+'-'+NewLabel+'__'+NewFormat+'_1.root '+NewRelease+'/'+NewTag+'/'+sample+'/'+'val.'+sample+'.root')
         elif (os.path.isfile(newSample)) :
             os.system('cp '+newSample+' '+NewRelease+'/'+NewTag+'/'+sample)
             
         if (os.path.isfile(RefRelease+'/'+RefTag+'/'+sample+'/val'+sample+'.root')!=True and os.path.isfile(refSample)) :
             os.system('ln -s '+refSample+' '+RefRelease+'/'+RefTag+'/'+sample)

         cfgFileName=sample+'_'+NewRelease+'_'+RefRelease
         hltcfgFileName='HLT'+sample+'_'+NewRelease+'_'+RefRelease

         if os.path.isfile(refSample ):
             replace_map_RECO = { 'DATATYPE': 'RECO', 'NEW_FILE':NewRelease+'/'+NewTag+'/'+sample+'/val.'+sample+'.root', 'REF_FILE':RefRelease+'/'+RefTag+'/'+sample+'/val.'+sample+'.root', 'REF_LABEL':sample, 'NEW_LABEL': sample, 'REF_RELEASE':RefRelease, 'NEW_RELEASE':NewRelease, 'REFSELECTION':RefTag, 'NEWSELECTION':NewTag, 'TrackValHistoPublisher': cfgFileName}
             if (ValidateHLT):
                 replace_map_HLT = { 'DATATYPE': 'HLT', 'NEW_FILE':NewRelease+'/'+NewTag+'/'+sample+'/val.'+sample+'.root', 'REF_FILE':RefRelease+'/'+RefTag+'/'+sample+'/val.'+sample+'.root', 'REF_LABEL':sample, 'NEW_LABEL': sample, 'REF_RELEASE':RefRelease, 'NEW_RELEASE':NewRelease, 'REFSELECTION':RefTag, 'NEWSELECTION':NewTag, 'TrackValHistoPublisher': hltcfgFileName}
         else:
             print "No reference file found at: ", RefRelease+'/'+RefTag
             replace_map_RECO = { 'DATATYPE': 'RECO', 'NEW_FILE':NewRelease+'/'+NewTag+'/'+sample+'/val.'+sample+'.root', 'REF_FILE':NewRelease+'/'+NewTag+'/'+sample+'/val.'+sample+'.root', 'REF_LABEL':sample, 'NEW_LABEL': sample, 'REF_RELEASE':NewRelease, 'NEW_RELEASE':NewRelease, 'REFSELECTION':NewTag, 'NEWSELECTION':NewTag, 'TrackValHistoPublisher': cfgFileName}
             if (ValidateHLT):
                 replace_map_HLT = { 'DATATYPE': 'HLT', 'NEW_FILE':NewRelease+'/'+NewTag+'/'+sample+'/val.'+sample+'.root', 'REF_FILE':NewRelease+'/'+NewTag+'/'+sample+'/val.'+sample+'.root', 'REF_LABEL':sample, 'NEW_LABEL': sample, 'REF_RELEASE':NewRelease, 'NEW_RELEASE':NewRelease, 'REFSELECTION':NewTag, 'NEWSELECTION':NewTag, 'TrackValHistoPublisher': hltcfgFileName}

         templatemacroFile = open(macro, 'r')
         macroFile = open(cfgFileName+'.C' , 'w' )
         replace(replace_map_RECO, templatemacroFile, macroFile)

         if (ValidateHLT):
             templatemacroFile = open(macro, 'r')
             hltmacroFile = open(hltcfgFileName+'.C' , 'w' )
             replace(replace_map_HLT, templatemacroFile, hltmacroFile)

         if(Submit):
             os.system('root -b -q -l '+ cfgFileName+'.C'+ '>  macro.'+cfgFileName+'.log')
             if (ValidateHLT):
                 os.system('root -b -q -l '+ hltcfgFileName+'.C'+ '>  macro.'+hltcfgFileName+'.log')

         if(Publish):
             if(os.path.exists(newdir)==False):
                 os.makedirs(newdir)
             os.system('rm '+NewRelease+'/'+NewTag+'/'+sample+'/val.'+sample+'.root')  
             os.system('scp -r '+NewRelease+'/'+NewTag+'/'+sample+'/* ' + newdir)
