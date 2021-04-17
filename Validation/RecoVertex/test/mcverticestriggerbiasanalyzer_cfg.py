import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

from PhysicsTools.PatAlgos.tools.helpers import cloneProcessingSnippet

process = cms.Process("MCverticesTriggerBias")

#prepare options

options = VarParsing.VarParsing("analysis")

options.register ('globalTag',
                  "DONOTEXIST",
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "GlobalTag")
options.register ('HLTprocess',
                  "HLT",
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "HLTProcess")
options.register ('L1Collection',
                  "gtDigis",
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "L1DigiCollection")
options.register ('triggerPaths',
                  "",
                  VarParsing.VarParsing.multiplicity.list, # singleton or list
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "list of HLT paths")
options.register ('triggerLabels',
                  "",
                  VarParsing.VarParsing.multiplicity.list, # singleton or list
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "list of labels")
options.register ('negateFlags',
                  "",
                  VarParsing.VarParsing.multiplicity.list, # singleton or list
                  VarParsing.VarParsing.varType.int,          # string, int, or float
                  "list of flags to negate HLT selection")

options.parseArguments()

#
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True),
    fileMode = cms.untracked.string("FULLMERGE")
    )

process.load("FWCore.MessageService.MessageLogger_cfi")

process.MessageLogger.cout.enable = cms.untracked.bool(True)
process.MessageLogger.cout.threshold = cms.untracked.string("INFO")
process.MessageLogger.cout.default = cms.untracked.PSet(
    limit = cms.untracked.int32(10000000)
    )
process.MessageLogger.cout.FwkReport = cms.untracked.PSet(
    reportEvery = cms.untracked.int32(10000)
    )

process.MessageLogger.cerr.enable = cms.untracked.bool(True)
process.MessageLogger.cerr.threshold = cms.untracked.string("WARNING")
process.MessageLogger.cerr.default = cms.untracked.PSet(
    limit = cms.untracked.int32(10000000)
    )
process.MessageLogger.cerr.FwkReport = cms.untracked.PSet(
    reportEvery = cms.untracked.int32(100000)
    )

#------------------------------------------------------------------

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = cms.Source("PoolSource",
                    fileNames = cms.untracked.vstring(),
    
#                    skipBadFiles = cms.untracked.bool(True),
                    inputCommands = cms.untracked.vstring("keep *", "drop *_MEtoEDMConverter_*_*")
                    )


process.source.fileNames = cms.untracked.vstring(options.inputFiles)

#---------------------------------------------------------------------------------------------------
# Trigger Selection

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import *
process.hltSelection = triggerResultsFilter.clone(
                                                  triggerConditions = cms.vstring("HLT_*"),
                                                  hltResults = cms.InputTag( "TriggerResults", "", options.HLTprocess ),
                                                  l1tResults = cms.InputTag( options.L1Collection ),
                                                  throw = cms.bool(False)
                                                  )

#---------------------------------------------------------------------------------------------------

process.load("Validation.RecoVertex.mcverticesanalyzer_cfi")


process.seqComplete= cms.Sequence(process.hltSelection +  process.mcverticesanalyzer) 

for label, trigger,negate in zip(options.triggerLabels,options.triggerPaths,options.negateFlags):
   cloneProcessingSnippet(process,process.seqComplete,label)
   getattr(process,"hltSelection"+label).triggerConditions = cms.vstring(trigger)

   if negate == 1:
      tempmodule = getattr(process,"hltSelection"+label)
      getattr(process,"seqComplete"+label).replace(getattr(process,"hltSelection"+label),~tempmodule)

   setattr(process,"p"+label,cms.Path(getattr(process,"seqComplete"+label))) 

#----GlobalTag ------------------------

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, options.globalTag, '')

process.TFileService = cms.Service('TFileService',
                                   fileName = cms.string('mcverticestriggerbias.root')
                                   )

