#!/usr/bin/env cmsRun

import shutil
import sys

from Validation.RecoTau.ValidationOptions_cff import *
import Validation.RecoTau.RecoTauValidation_cfi as validation

process = cms.Process("TEST")

# command options defined in Validation/RecoTau/python/ValidationOptions_cfi
options.parseArguments()

checkOptionsForBadInput()

if not calledBycmsRun() and not options.gridJob:
   print "Run 'cmsRun RunTauValidation_cfg.py help' for options."
   # quit here so we dont' create a bunch of directories
   #  if the user only wants the help
   #sys.exit()

# Make sure we dont' clobber another directory! Skip in batch mode (runs from an LSF machine)
if not CMSSWEnvironmentIsCurrent() and options.batchNumber == -1 and not options.gridJob:
   print "CMSSW_BASE points to a different directory, please rerun cmsenv!"
   sys.exit()


# DQM store, PDT sources etc
process.load("Configuration.StandardSequences.Services_cff")

######################################
#                                    #
#       Output Info Store            #
#                                    #
######################################

"""
   Data is stored in

   TauID/[EventType]_[DataSource]_[Conditions][label]

"""

#outputDirName = "Validation_%s" % ReleaseVersion
outputDirName = "TauID"


outputDir = os.path.join(os.getcwd(), outputDirName) 
# This is the directory where we store the stuff about our current configuration
outputBaseDir = outputDir

subDirName = ""

subDirName += "%s_%s" % (options.eventType, options.dataSource)

if options.conditions != "whatever":
   subDirName += "_%s" % options.conditions.replace('::', '_')

if (options.label != "none"):
   subDirName += "_" + options.label

outputDir = os.path.join(outputDir, subDirName)

# Store configuration, showtags, etc in a sub directory
configDir = os.path.join(outputDir, "Config")

if os.path.exists(outputDir) and options.batchNumber < 0:# and not options.gridJob:
   print "Output directory %s already exists!  OK to overwrite?" % outputDir
   while True:
      input = raw_input("Please enter [y/n] ")
      if (input == 'y'):
         break
      elif (input == 'n'):
         print " ...exiting."
         sys.exit()

if not os.path.exists(outputDir):
   os.makedirs(outputDir)

if not os.path.exists(configDir):
   os.makedirs(configDir)

######################################
#                                    #
#       Data Source Setup            #
#                                    #
######################################

def LoadDataCffFile(theFile):
   outputFileName = os.path.join(configDir, "DataSource_cff.py")
   process.load(theFile)
   outputFile = open(outputFileName,'w')
   outputFile.write('import FWCore.ParameterSet.Config as cms\n')
   outputFile.write('source = %s\n'%process.source)

process.schedule = cms.Schedule()

# Check if we are simulating events - if so we need to define our generator
if options.dataSource.find('sim') != -1:
   if options.eventType == "ZTT":
      process.load("Configuration.Generator.ZTT_Tauola_All_hadronic_cfi")
   elif options.eventType == "QCD":
      process.load("Configuration.Generator.QCDForPF_cfi")

# Run on a RECO (eg RelVal)
if options.dataSource.find('recoFiles') != -1:
   myFile = options.sourceFile
   if myFile == 'none':
      myFile = "Validation.RecoTau.sources.EventSource_%s_RECO_cff" % options.eventType
      #myFile = os.path.join(ReleaseBase, "Validation/RecoTau/test", "EventSource_%s_RECO_cff.py" % options.eventType)
   LoadDataCffFile(myFile)
   if len(process.source.fileNames) == 0 and not options.gridJob:
      import Validation.RecoTau.DBSApi_cff as mydbs
      if os.path.isfile('SourcesDatabase.xml'):
         print "Trying to retrieve the input files from SourcesDatabase.xml..."
         xml = open('SourcesDatabase.xml','r')
         mydbs.loadXML(xml,options.eventType,process.source)
      if len(process.source.fileNames) == 0:
         print "Accessing DBS to retrieve the input files..."
         mydbs.FillSource(options.eventType,process.source)
      if len(process.source.fileNames) == 0:
         sys.exit(0)
      print process.source
   # check if we want to rerun PFTau
   if options.dataSource.find('PFTau') != -1:
      process.load("Configuration.StandardSequences.Geometry_cff")
      process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
      process.load("Configuration.StandardSequences.MagneticField_cff")
      process.load("RecoTauTag.Configuration.RecoPFTauTag_cff")
      process.runPFTau = cms.Path(process.PFTau)
      process.schedule.append(process.runPFTau)
   if options.dataSource.find('CaloTau') != -1:
      process.load("Configuration.StandardSequences.Geometry_cff")
      process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
      process.load("Configuration.StandardSequences.MagneticField_cff")
      process.load("RecoTauTag.Configuration.RecoTauTag_cff")
      process.runCaloTau = cms.Path(process.tautagging)
      process.schedule.append(process.runCaloTau)

# Run on DIGI files and re-RECO
elif options.dataSource == 'digiFiles':
   myFile = options.sourceFile
   if myFile == 'none':
      myFile = "Validation.RecoTau.EventSource_%s_DIGI_cff" % options.eventType
      #myFile = os.path.join(ReleaseBase, "Validation/RecoTau/test", "EventSource_%s_DIGI_cff.py" % options.eventType)
   LoadDataCffFile(myFile)
   # get the sequences need to redo RECO
   process.load("Validation.RecoTau.ProduceTausFromDigis_cff")
   process.makeTausFromDigiFiles = cms.Path(proces.makeTausFromDigis)
   process.schedule.append(process.makeTausFromDigiFiles)

# Generate FASTSIM DATA
elif options.dataSource == 'fastsim':
   process.load("Validation.RecoTau.ProduceTausWithFastSim_cff")
   process.fastSimTaus = cms.Path(process.makeTausWithFastSim)
   process.schedule.append(process.fastSimTaus)

# Generate FULLSIM DATA
elif options.dataSource == 'fullsim':
   process.load("Validation.RecoTau.ProduceFullSimAndDigisForTaus_cff")
   process.load("Validation.RecoTau.ProduceTausFromDigis_cff")
   process.fullSimTaus = cms.Path(process.simAndDigitizeForTaus*process.makeTausFromDigis)
   process.schedule.append(process.fullSimTaus)

# Specify conditions if desired
if options.conditions != "whatever":
   process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
   process.GlobalTag.globaltag = options.conditions

# have to set max events here, since it may get written by the 
# dataSource cffs
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(options.maxEvents)
)

# Skip events, if we are running in batch mode on files
if options.batchNumber >= 0 and options.dataSource.find('Files') != -1:
   process.source.skipEvents = cms.untracked.uint32(options.batchNumber*options.maxEvents)

######################################
#                                    #
#       Validation Setup             #
#                                    #
######################################

# Store the tags and CVS diff to the tags, and the current release
#  only do this once in a batch job.  The additional tar file is a fail safe - 
#  the parameters shouldn't change in outputDir.
if (options.batchNumber <= 0 ):#and not options.gridJob):
   os.system("cd $CMSSW_BASE/src; \
              showtags -t -r > showtags.txt; \
              cvs -q diff >& diffToTags.patch;\
              cvs -q diff -r %s >& diffToVanillaRelease.patch; \
              tar -cvzf TagsAndDiff.tar.gz showtags.txt *.patch; \
              mv showtags.txt *.patch %s; \
              mv TagsAndDiff.tar.gz %s" % (ReleaseVersion, configDir, configDir))

if options.batchNumber >= 0:
   # store the batch produced root files in a sub directory
   outputDir = os.path.join(outputDir, "BatchJobs")
   if not os.path.exists(outputDir):
      os.mkdir(outputDir)

#Validation output file
outputFileNameBase = "TauVal_%s" % ReleaseVersion
if options.label != "none":
   outputFileNameBase += "_%s" % options.label
outputFileNameBase += "_"
outputFileNameBase += options.eventType

if options.batchNumber >= 0:
   outputFileNameBase += "_%i" % options.batchNumber
   options.writeEDMFile = options.writeEDMFile.replace(".root", "_%i.root" % options.batchNumber)
outputFileNameBase += ".root"

if validation.StandardMatchingParameters.recoCuts.value() != "" and validation.StandardMatchingParameters.genCuts.value() != "":
  print 'Matching: cut(s) set to: reco "%s", gen "%s".' % (validation.StandardMatchingParameters.recoCuts.value(), validation.StandardMatchingParameters.genCuts.value())
else:
  if validation.StandardMatchingParameters.recoCuts.value() != "":
    print 'Matching: reco cut(s) set to: "%s".' % validation.StandardMatchingParameters.recoCuts.value()
  if validation.StandardMatchingParameters.genCuts.value() != "":
    print 'Matching: gen cut(s) set to: "%s".' % validation.StandardMatchingParameters.genCuts.value()

if options.gridJob:
   outputFileName = 'TauVal_GridJob.root'
else:
   outputFileName = os.path.join(outputDir, outputFileNameBase)

print 'The output file will be: '+outputFileName
if options.gridJob:
   cfg=open('./crab.cfg', 'r')
   cfgContent=cfg.read()
   if cfgContent.find(outputFileName) == -1:
      print "ERROR: CRAB output file not matching the grid one!\nexiting..."
      sys.exit()

process.saveTauEff = cms.EDAnalyzer("TauDQMSimpleFileSaver",
  outputFileName = cms.string(outputFileName)
)

process.load("Validation.RecoTau.dataTypes.ValidateTausOn%s_cff" % options.eventType)
process.validation = cms.Path( process.produceDenominator )#getattr(process,'produceDenominator'+options.eventType) )

if options.batchNumber >= 0 or options.gridJob:
   process.validation *= process.runTauValidationBatchMode #in batch mode, the efficiencies are not computed - only the num/denom
else:
   process.validation *= process.runTauValidation

process.validation *= process.saveTauEff #save the output

process.schedule.append(process.validation)

if options.batchNumber >= 0:
   newSeed = process.RandomNumberGeneratorService.theSource.initialSeed.value() + options.batchNumber 
   process.RandomNumberGeneratorService.theSource.initialSeed = cms.untracked.uint32(newSeed)
   process.RandomNumberGeneratorService.generator.initialSeed = cms.untracked.uint32(newSeed)
   print "I'm setting the random seed to ", newSeed


process.load("RecoTauTag.Configuration.RecoTauTag_EventContent_cff")

TauTagValOutputCommands = cms.PSet(
      outputCommands = cms.untracked.vstring('drop *',
         'keep recoPFCandidates_*_*_*',
         'keep *_genParticles*_*_*',
         'keep *_ak5GenJets_*_*',
         'keep *_tauGenJets*_*_*',
         'keep *_selectedGenTauDecays*_*_*'
         )
      )

TauTagValOutputCommands.outputCommands.extend(process.RecoTauTagRECO.outputCommands)

# talk to output module
if options.writeEDMFile != "":
   # Find where the EDM file should be written.  This is set by the 
   #  to the working directory when running jobs on lxbatch
   try:
      edmOutputDir = os.environ['edmOutputDir']
      options.writeEDMFile = os.path.join(edmOutputDir, options.writeEDMFile)
   except KeyError:
      pass

   process.out = cms.OutputModule("PoolOutputModule",
         TauTagValOutputCommands,
         verbose = cms.untracked.bool(False),
         fileName = cms.untracked.string (options.writeEDMFile)
   )
   myOutpath = cms.EndPath(process.out)
   process.schedule.append(myOutpath)

if options.myModifications != ['none']:
   for aModifier in options.myModifications:
      process.load(aModifier.replace('.py',''))


######################################
#                                    #
#       CFG dump                     #
#                                    #
######################################
#process.Timing = cms.Service("Timing",
#         useJobReport = cms.untracked.bool(True)
#	 )
#process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck",
#         useJobReport = cms.untracked.bool(True)
#	 )

#if grid job end here
if not options.gridJob:

   dumpFileName = "cfgDump"
   if options.batchNumber >= 0:
      dumpFileName += "_"
      dumpFileName += str(options.batchNumber)
      
   dumpFileName += ".py"
   
   processDumpFile = open('%s/%s' % (configDir, dumpFileName), 'w')
   
   print >> processDumpFile, process.dumpPython()

