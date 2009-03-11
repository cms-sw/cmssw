#! /usr/bin/env python

import os
import sys
import fileinput
import string

NewRelease='CMSSW_3_1_0_pre3'
RefRelease='CMSSW_3_1_0_pre2'

samples= ['RelValSingleMuPt10','RelValSingleMuPt100','RelValSingleMuPt1000','RelValTTbar']

Submit=True
Publish=False

NewTag='IDEAL'
RefTag='IDEAL'

NewLabel='IDEAL_30X'
RefLabel='IDEAL_30X'

#ReferenceSelection='IDEAL_V5_noPU'
#StartupReferenceSelection='STARTUP_V4_noPU'

RefRepository = '/afs/cern.ch/cms/Physics/muon/CMSSW/Performance/RecoMuon/Validation/val'
NewRepository = '/afs/cern.ch/cms/Physics/muon/CMSSW/Performance/RecoMuon/Validation/val'


macro='macro/TrackValHistoPublisher.C'
hltmacro='macro/HLTTrackValHistoPublisher.C'

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
     templatemacroFile = open(macro, 'r')
     templatehltmacroFile = open(hltmacro, 'r')

     newdir=NewRepository+'/'+NewRelease+'/'+NewTag+'/'+sample 



     if(os.path.exists(NewRelease+'/'+NewTag+'/'+sample)==False):
         os.makedirs(NewRelease+'/'+NewTag+'/'+sample)

     if(os.path.exists(RefRelease+'/'+RefTag+'/'+sample)==False):
         os.makedirs(RefRelease+'/'+RefTag+'/'+sample)


     if(os.path.isfile(NewRelease+'/'+NewTag+'/'+sample+'/building.pdf')!=True):
         newSample=NewRepository+'/'+NewRelease+'/'+NewTag+'/'+sample+'/'+'val.'+sample+'.root'
         refSample=RefRepository+'/'+RefRelease+'/'+RefTag+'/'+sample+'/'+'val.'+sample+'.root'
         if (os.path.isfile(NewRelease+'/'+NewTag+'/'+sample+'/val'+sample+'.root')==False and os.path.isfile(newSample)) :
             os.system('cp '+newSample+' '+NewRelease+'/'+NewTag+'/'+sample)
         if (os.path.isfile(RefRelease+'/'+RefTag+'/'+sample+'/val'+sample+'.root')==False and os.path.isfile(refSample)) :
             os.system('cp '+refSample+' '+RefRelease+'/'+RefTag+'/'+sample)

         cfgFileName=sample+'_'+NewRelease+'_'+RefRelease
         hltcfgFileName='HLT'+sample+'_'+NewRelease+'_'+RefRelease

         if os.path.isfile(refSample ):
             replace_map = { 'NEW_FILE':NewRelease+'/'+NewTag+'/'+sample+'/val.'+sample+'.root', 'REF_FILE':RefRelease+'/'+RefTag+'/'+sample+'/val.'+sample+'.root', 'REF_LABEL':sample, 'NEW_LABEL': sample, 'REF_RELEASE':RefRelease, 'NEW_RELEASE':NewRelease, 'REFSELECTION':RefTag, 'NEWSELECTION':NewTag, 'TrackValHistoPublisher': cfgFileName}
         else:
             print "No reference file found at: ", RefRelease+'/'+RefTag
             replace_map = { 'NEW_FILE':NewRelease+'/'+NewTag+'/'+sample+'/val.'+sample+'.root', 'REF_FILE':NewRelease+'/'+NewTag+'/'+sample+'/val.'+sample+'.root', 'REF_LABEL':sample, 'NEW_LABEL': sample, 'REF_RELEASE':NewRelease, 'NEW_RELEASE':NewRelease, 'REFSELECTION':NewTag, 'NEWSELECTION':NewTag, 'TrackValHistoPublisher': cfgFileName}

         macroFile = open(cfgFileName+'.C' , 'w' )
         replace(replace_map, templatemacroFile, macroFile)

         hltmacroFile = open(hltcfgFileName+'.C' , 'w' )
         replace(replace_map, templatehltmacroFile, hltmacroFile)

         if(Submit):
             os.system('root -b -q -l '+ cfgFileName+'.C'+ '>  macro.'+cfgFileName+'.log')
             os.system('root -b -q -l '+ hltcfgFileName+'.C'+ '>  macro.'+hltcfgFileName+'.log')

         if(Publish):
             if(os.path.exists(newdir)==False):
                 os.makedirs(newdir)
             os.system('rm '+NewRelease+'/'+NewTag+'/'+sample+'/val.'+sample+'.root')  
             os.system('scp -r '+NewRelease+'/'+NewTag+'/'+sample+'/* ' + newdir)
