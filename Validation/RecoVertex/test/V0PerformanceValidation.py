#! /usr/bin/env python

import os
import sys
import fileinput
import string

# Reference release
RefRelease='CMSSW_3_6_0_pre1'

# RelVal release (only set if different from $CMSSW_VERSION)
NewRelease='CMSSW_3_6_0_pre1'

# Startup and ideal samples list
#startupsamples = ['RelValTTbar', 'RelValMinBias', 'RelValQCD_Pt_3000_3500', 'RelValQCD_Pt_80_120']

# This is a pileup sample
#startupsamples = ['RelValTTbar_Tauola']
# This is the full set of STARTUP samples, to be used for timing studies
#startupsamples = ['RelValQCD_Pt_3000_3500', 'RelValTTbar', 'RelValMinBias']
#startupsamples = ['RelValQCD_Pt_3000_3500', 'RelValMinBias']
#startupsamples = ['RelValTTbar']
startupsamples = []

#$startupsamples = ['RelValQCD_Pt_80_120']

# MC relvals
#idealsamples = ['RelValSingleMuPt1', 'RelValSingleMuPt10', 'RelValSingleMuPt100', RelValSinglePiPt1']

# Pileup MC sample
#idealsamples = ['RelValZmumuJets_Pt_20_300_GEN']
idealsamples = []

# New samples array for 3_X_X
mcsamples = ['RelValTTbar']
#mcsamples = []

Sequence = 'only_validation'
#Sequence = 'harvesting'

# GlobalTags
IdealTag = 'DESIGN_3X_V21'
#StartupTag = 'START3X_V21'
StartupTag = 'STARTUP3X_V8H'
# This tag is the new replacement (as of 31X) for MC, IDEAL becomes DESIGN
# MC_31X_V9 works for CMSSW_3_4_0_preX
MCTag = 'MC_3XY_V21'
#MCTag = 'MC_3XY_V12'

# PileUp: PU, No PileUp: noPU
PileUp = 'noPU'

# Reference directory name
ReferenceSelection = 'DESIGN_3X_V21_'+PileUp
MCReferenceSelection = 'MC_3XY_V21_'+PileUp
StartupReferenceSelection = 'STARTUP3X_V8H_'+PileUp

# This is where the reference samples are stored
#RefRepository = '/afs/cern.ch/cms/performance/'
RefRepository = '/nfs/data35/cms/drell/val360pre1'
# NewRepository contains the files for the new release to be tested
NewRepository = '/nfs/data35/cms/drell/val360pre1'

# Default number of events
defaultNevents = '-1'

# Specify the number of events to be processed for specific samples (numbers must be strings)
Events = {}

# template file names, shouldn't need to be changed
cfg = 'v0PerformanceValidation_cfg.py'
macro = 'macros/V0ValHistoPublisher.C'

#This config is only for testing V0Producer changes.
#cfg = 'v0PerformanceValidation_v0prodtest_cfg.py'
#cfg = 'v0PerformanceValidation_origV0Prod_cfg.py'

###################################
### End configurable parameters ###
###################################

#----------------------------------------------------------------------------

#################
### Functions ###
#################

# Does replacement of strings in files
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


# This function does most of the work
def do_validation(samples, GlobalTag):
    global Sequence, RefSelection, RefRepository, NewSelection, NewRepository, defaultNevents, Events
    global cfg, macro, Tracksname
    #print 'Tag: ' + GlobalTag
    NewSelection = GlobalTag + '_' + PileUp

    for sample in samples:
        templateCfgFile = open(cfg, 'r')
        templateMacroFile = open(macro, 'r')
        newdir = NewRepository + '/' + NewRelease + '/' + NewSelection + '/' + sample
        cfgFileName = sample + GlobalTag

        if(os.path.isfile(newdir + '/K0sEff.pdf') != True):
            # If the job is harvesting, check if the file is already harvested
            harvestedfile = './DQM_V0001_R000000001__' + GlobalTag + '__' + sample + '__Validation.root'
            if(( Sequence == "harvesting" and os.path.isfile(harvestedfile)) == False):
#                cmd = 'dbsql "find dataset.createdate, dataset where dataset like *'
		cmd = 'dbsql "find dataset where dataset like *'
                cmd += sample + '/' + NewRelease + '_' + GlobalTag + '*GEN-SIM-RECO*" '
                cmd += '| grep ' + sample + ' | grep -v test | sort | tail -1 | cut -f2 '
                print cmd
                print 'Finding data set...\n'
                dataset = os.popen(cmd).readline().strip()
                print 'DataSet: ', dataset, '\n'
                if dataset != "":
                    cmd2 = 'dbsql "find file where dataset like ' + dataset + '" | grep ' + sample
                    filenames = 'import FWCore.ParameterSet.Config as cms\n'
                    filenames += 'readFiles = cms.untracked.vstring()\n'
                    filenames += 'secFiles = cms.untracked.vstring()\n'
                    filenames += 'source = cms.Source("PoolSource", fileNames = readFiles, secondaryFileNames = secFiles)\n'
                    filenames += 'readFiles.extend([\n'
                    first = True
                    #print cmd2
                    print 'Getting file names for dataset ', dataset, '\n'
                    for line in os.popen(cmd2).readlines():
                        filename = line.strip()
                        if first == True:
                            filenames += "    '"
                            filenames += filename
                            filenames += "'"
                            first = False
                        else:
                            filenames += ",\n    '"
                            filenames += filename
                            filenames += "'"
                    filenames += ']);\n'
                    # if not harvesting, find secondary file names
                    if(Sequence != 'harvesting'):
                        cmd3 = 'dbsql "find dataset.parent where dataset like ' + dataset + '" | grep ' + sample
			#print cmd3
                        print 'Getting parent dataset...\n'
                        parentdataset = os.popen(cmd3).readline()
                        print 'Parent dataset: ', parentdataset, '\n'
                        if parentdataset != "":
                            cmd4 = 'dbsql "find file where dataset like ' + parentdataset + '" | grep ' + sample
                            #print 'cmd4', cmd4
                            filenames += 'secFiles.extend([\n'
                            first = True
                            for line in os.popen(cmd4).readlines():
                                secfilename = line.strip()
                                if first == True:
                                    filenames += "    '"
                                    filenames += secfilename
                                    filenames += "'"
                                    first = False
                                else:
                                    filenames += ",\n    '"
                                    filenames += secfilename
                                    filenames += "'"
                            filenames += ']);\n\n'
                        else:
                            print "No primary dataset found, skipping sample ", sample
                            continue
                    else:
                        filenames += 'secFiles.extend((     ))'

                    cfgFile = open(cfgFileName + '.py', 'w')
                    cfgFile.write(filenames)

                    if(sample in Events):
                        Nevents = Events[sample]
                    else:
                        Nevents = defaultNevents

                    symbol_map = {'NEVENT':Nevents, 'GLOBALTAG':GlobalTag, 'SEQUENCE':Sequence, 'SAMPLE':sample}
                    cfgfile = open(cfgFileName + '.py', 'a')
                    replace(symbol_map, templateCfgFile, cfgFile)

                    cmdrun = 'cmsRun ' + cfgFileName + '.py >& ' + cfgFileName + '.log < /dev/zero'
                    print 'Running validation on sample ' + sample + '...\n'
                    #print cmdrun
                    retcode = os.system(cmdrun)
                else:
                    print 'No dataset found, skipping sample ', sample, '\n'
                    continue
            else:
                retcode = 0
            if(retcode):
                print 'Job for sample ' + sample + ' failed.\n'
                print 'Check log file ' + sample + '.log for details.\n'
            else:
                if Sequence == 'harvesting':
                    rootcommand = 'root -b -q -l CopySubdir.C\\('+'\\\"DQM_V0001_R000000001__'+GlobalTag+'__'+sample+'__Validation.root\\\", \\\"val.'+sample+'.root\\\"\\) >& /dev/null'
                    os.system(rootcommand)
                referenceSample = RefRepository + '/' + RefRelease + '/' + RefSelection + '/' + sample + '/' + 'val.'+sample+'.root'
                print 'Reference sample: ' + referenceSample
            
                if os.path.isfile(referenceSample):
                    print 'Found reference.'
                    replace_map = {'NEW_FILE':'val.'+sample+'.root', 'REF_FILE':RefRelease+'/'+RefSelection+'/'+'val.'+sample+'.root', 'REF_LABEL':sample, 'NEW_LABEL':sample, 'REF_RELEASE':RefRelease, 'NEW_RELEASE':NewRelease, 'REFSELECTION':RefSelection, 'NEWSELECTION':NewSelection, 'V0ValHistoPublisher':cfgFileName }
                
                    if not os.path.exists(RefRelease + '/' + RefSelection):
                        os.makedirs(RefRelease + '/' + RefSelection)
                        os.system('cp ' + referenceSample + ' ' + RefRelease+'/'+RefSelection)
                
                else:
                    print "No reference file found at ", RefRepository+'/'+RefRelease+'/'+RefSelection
                    replace_map = {'NEW_FILE':'val.'+sample+'.root', 'REF_FILE':'val.'+sample+'.root', 'REF_LABEL':sample, 'NEW_LABEL':sample, 'REF_RELEASE':NewRelease, 'NEW_RELEASE':NewRelease, 'REFSELECTION':NewSelection, 'NEWSELECTION':NewSelection, 'V0ValHistoPublisher':cfgFileName }

                macroFile = open(cfgFileName+'.C', 'w')
                replace(replace_map, templateMacroFile, macroFile)

                os.system('root -b -q -l ' + cfgFileName + '.C' + ' > macro.' + cfgFileName + '.log')

                if not os.path.exists(newdir):
                    os.makedirs(newdir)

                print "Moving PDF and PNG files for sample: ", sample
                os.system('mv *pdf *png ' + newdir)

                print "Moving ROOT file for sample: ", sample
                os.system('mv val.'+sample+'.root ' + newdir)

                print "Copying .py file for sample: ", sample
                os.system('cp '+cfgFileName+'.py ' + newdir)
        else:
            print 'Validation for sample '+sample+' already done, skipping.\n'

##############################################################################
### Main part of script, this runs do_validation for all specified samples ###
##############################################################################
if(NewRelease == ''):
    try:
        NewRelease = os.environ["CMSSW_VERSION"]
    except KeyError:
        print >>sys.stderr, 'Error: The environment variable CMSSW_VERSION is not set.'
        print >>sys.stderr, '       Please run cmsenv in the appropriate release area.'
else:
    try:
        os.environ["CMSSW_VERSION"]
    except KeyError:
        print >>sys.stderr, 'Error: CMSSW environment variables are not set.'
        print >>sys.stderr, '       Please run cmsenv in the appropriate release area.'

NewSelection = ''
RefSelection = MCReferenceSelection
do_validation(mcsamples, MCTag)

NewSelection = ''
RefSelection = StartupReferenceSelection
do_validation(startupsamples, StartupTag)

NewSelection = ''
RefSelection = ReferenceSelection
do_validation(idealsamples, IdealTag)

print 'Finished running validation.'
