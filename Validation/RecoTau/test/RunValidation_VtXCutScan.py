#!/usr/bin/env cmsRun

import shutil
import sys

from Validation.RecoTau.ValidationOptions_cff import *

process = cms.Process("TEST")

# command options defined in Validation/RecoTau/python/ValidationOptions_cfi
options.parseArguments()

checkOptionsForBadInput()

## if not calledBycmsRun() and not options.gridJob:
##    print "Run 'cmsRun RunTauValidation_cfg.py help' for options."
##    # quit here so we dont' create a bunch of directories
##    #  if the user only wants the help
##    sys.exit()

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
   if not os.path.isfile(theFile):
      print "Error - %s is not a file!" % theFile
      sys.exit()
   outputFile = os.path.join(configDir, "DataSource_cff.py")
   shutil.copy(theFile, outputFile)
   process.load(theFile.replace(".py", ""))

myFile = options.sourceFile
if myFile == 'none':
   myFile = "EventSource_%s_RECO_cff.py" % options.eventType
   #myFile = os.path.join(ReleaseBase, "Validation/RecoTau/test", "EventSource_%s_RECO_cff.py" % options.eventType)
LoadDataCffFile(myFile)
#Reruns PFTau
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("RecoTauTag.Configuration.RecoPFTauTag_cff")
process.hpsSequence = cms.Sequence( process.recoTauCommonSequence*process.recoTauClassicHPSSequence )

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
outputFileNameBase += "_DBScan.root"

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

process.load("Validation.RecoTau.ValidateTausOn%s_cff" % options.eventType)

#Sets the cuts to what defined in 2/5/11 meeting
process.hpsPFTauDiscriminationByVLooseIsolation.qualityCuts.pvFindingAlgo = 'highestWeightForLeadTrack'
process.hpsPFTauDiscriminationByVLooseIsolation.ApplyDiscriminationByECALIsolation = False
process.hpsPFTauDiscriminationByVLooseIsolation.applyDeltaBetaCorrection = False
process.hpsPFTauDiscriminationByVLooseIsolation.qualityCuts.signalQualityCuts.minTrackVertexWeight = -1
process.hpsPFTauDiscriminationByVLooseIsolation.qualityCuts.isolationQualityCuts.minTrackVertexWeight = -1

process.hpsPFTauDiscriminationByMediumIsolation.qualityCuts.pvFindingAlgo = 'highestWeightForLeadTrack'
process.hpsPFTauDiscriminationByMediumIsolation.ApplyDiscriminationByECALIsolation = False
process.hpsPFTauDiscriminationByMediumIsolation.applyDeltaBetaCorrection = False
process.hpsPFTauDiscriminationByMediumIsolation.qualityCuts.signalQualityCuts.minTrackVertexWeight = -1
process.hpsPFTauDiscriminationByMediumIsolation.qualityCuts.isolationQualityCuts.minTrackVertexWeight = -1

process.hpsPFTauDiscriminationByLooseIsolation.qualityCuts.pvFindingAlgo = 'highestWeightForLeadTrack'
process.hpsPFTauDiscriminationByLooseIsolation.ApplyDiscriminationByECALIsolation = False
process.hpsPFTauDiscriminationByLooseIsolation.applyDeltaBetaCorrection = False
process.hpsPFTauDiscriminationByLooseIsolation.qualityCuts.signalQualityCuts.minTrackVertexWeight = -1
process.hpsPFTauDiscriminationByLooseIsolation.qualityCuts.isolationQualityCuts.minTrackVertexWeight = -1

process.hpsPFTauDiscriminationByTightIsolation.qualityCuts.pvFindingAlgo = 'highestWeightForLeadTrack'
process.hpsPFTauDiscriminationByTightIsolation.ApplyDiscriminationByECALIsolation = False
process.hpsPFTauDiscriminationByTightIsolation.applyDeltaBetaCorrection = False
process.hpsPFTauDiscriminationByTightIsolation.qualityCuts.signalQualityCuts.minTrackVertexWeight = -1
process.hpsPFTauDiscriminationByTightIsolation.qualityCuts.isolationQualityCuts.minTrackVertexWeight = -1


process.preValidation = cms.Sequence(process.recoTauCommonSequence)

process.validation = cms.Sequence(
   process.ak5PFJetsLegacyHPSPiZeros *
   process.combinatoricRecoTaus *
   process.produceAndDiscriminateHPSPFTaus*
   process.produceDenominator *
   process.runTauValidationBatchMode #in batch mode, the efficiencies are not computed - only the num/denom
  )

import PhysicsTools.PatAlgos.tools.helpers as configtools

process.vtxStudy = cms.Sequence()

#---------------------------------------------------------------------------------
#               Cloning process to scan over several DzCuts
#---------------------------------------------------------------------------------
dzCuts = [0.05 ,0.10 , 0.15 , 0.2]
for dzCut in dzCuts:
   # Make a loose-DZ copy
   #print 'creating '+addedLabel
   process.hpsPFTauDiscriminationByVLooseIsolation.qualityCuts.signalQualityCuts.maxDeltaZ = dzCut
   process.hpsPFTauDiscriminationByVLooseIsolation.qualityCuts.isolationQualityCuts.maxDeltaZ = dzCut
   process.hpsPFTauDiscriminationByMediumIsolation.qualityCuts.signalQualityCuts.maxDeltaZ = dzCut
   process.hpsPFTauDiscriminationByMediumIsolation.qualityCuts.isolationQualityCuts.maxDeltaZ = dzCut
   process.hpsPFTauDiscriminationByLooseIsolation.qualityCuts.signalQualityCuts.maxDeltaZ = dzCut
   process.hpsPFTauDiscriminationByLooseIsolation.qualityCuts.isolationQualityCuts.maxDeltaZ = dzCut
   process.hpsPFTauDiscriminationByTightIsolation.qualityCuts.signalQualityCuts.maxDeltaZ = dzCut
   process.hpsPFTauDiscriminationByTightIsolation.qualityCuts.isolationQualityCuts.maxDeltaZ = dzCut

   addedLabel = 'DZCut%i'%(int(dzCut*100))
   configtools.cloneProcessingSnippet( process, process.validation, addedLabel)
   #checking we did everything correctly
   assert( hasattr(process,'validation%s'%(addedLabel) ) )
   assert( getattr(process,'hpsPFTauDiscriminationByVLooseIsolation%s'%(addedLabel) ).qualityCuts.signalQualityCuts.maxDeltaZ  == dzCut )
   assert( getattr(process,'hpsPFTauDiscriminationByMediumIsolation%s'%(addedLabel) ).qualityCuts.signalQualityCuts.maxDeltaZ  == dzCut )
   assert( getattr(process,'hpsPFTauDiscriminationByLooseIsolation%s'%(addedLabel) ).qualityCuts.signalQualityCuts.maxDeltaZ  == dzCut )
   assert( getattr(process,'hpsPFTauDiscriminationByTightIsolation%s'%(addedLabel) ).qualityCuts.signalQualityCuts.maxDeltaZ  == dzCut )
   process.vtxStudy += getattr(process,'validation%s'%(addedLabel) )
   assert( hasattr(process, 'RunHPSValidation%s'%(addedLabel))  )
   for entry in getattr(process, 'RunHPSValidation%s'%(addedLabel)).discriminators:
      entry.discriminator = entry.discriminator.value() + addedLabel
      #print addedLabel+' created'



#process.validation *= process.saveTauEff #save the output

process.preValPath = cms.Path(process.preValidation)
process.vtxPath = cms.Path(process.vtxStudy)
process.savePath = cms.Path(process.saveTauEff)

process.schedule = cms.Schedule()
process.schedule.append(process.preValPath)
process.schedule.append(process.vtxPath)
process.schedule.append(process.savePath)

process.load("RecoTauTag.Configuration.RecoTauTag_EventContent_cff")

TauTagValOutputCommands = cms.PSet(
      outputCommands = cms.untracked.vstring('drop *',
         'keep recoPFCandidates_*_*_*',
         'keep *_genParticles*_*_*',
         'keep *_iterativeCone5GenJets_*_*',
         'keep *_tauGenJets*_*_*',
         'keep *_selectedGenTauDecays*_*_*'
         )
      )

TauTagValOutputCommands.outputCommands.extend(process.RecoTauTagRECO.outputCommands)

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

processDumpFile = open('VtxTest.py','w')
print >> processDumpFile, process.dumpPython()
#if grid job end here
## if not options.gridJob:

##    dumpFileName = "cfgDump"
##    if options.batchNumber >= 0:
##       dumpFileName += "_"
##       dumpFileName += str(options.batchNumber)
      
##    dumpFileName += ".py"
   
##    processDumpFile = open('%s/%s' % (configDir, dumpFileName), 'w')
   
##    print >> processDumpFile, process.dumpPython()

