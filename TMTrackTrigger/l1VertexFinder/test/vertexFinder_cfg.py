#################################################################################################
# To run execute do
# cmsRun tmtt_tf_analysis_cfg.py Events=50 inputMC=Samples/Muons/PU0.txt histFile=outputHistFile.root
# where the arguments take default values if you don't specify them. You can change defaults below.
#################################################################################################

import FWCore.ParameterSet.Config as cms
import FWCore.Utilities.FileUtils as FileUtils
import FWCore.ParameterSet.VarParsing as VarParsing

process = cms.Process("L1TVertexFinder")

process.load('Configuration.Geometry.GeometryExtended2023D17Reco_cff')
process.load('Configuration.Geometry.GeometryExtended2023D17_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgradePLS3', '')
process.load("FWCore.MessageLogger.MessageLogger_cfi")


options = VarParsing.VarParsing ('analysis')

#--- Specify input MC
options.register('inputMC','ttbar_NoPU.txt', VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.string, "Files to be processed")

#--- Specify number of events to process.
options.register('Events',-1,VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.int,"Number of Events to analyze")

#--- Specify name of output histogram file.
options.register('histFile','Hist.root',VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.string,"Name of output histogram file")

options.parseArguments()


#--- input and output
list = FileUtils.loadListFromFile(options.inputMC)
readFiles = cms.untracked.vstring(*list)
secFiles = cms.untracked.vstring()

process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(False) )

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(options.Events) )

process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string(options.histFile)
)

process.source = cms.Source ("PoolSource",
                            fileNames = readFiles,
                            secondaryFileNames = secFiles,
                            # skipEvents = cms.untracked.uint32(500)
                            )


# process.out = cms.OutputModule("PoolOutputModule",
#     fileName = cms.untracked.string(options.outputFile),
#     outputCommands = cms.untracked.vstring(
#     	"keep *",
#     	"keep *_producer_*_*",
#     	"keep *_VertexProducer_*_*"
#     	)
# )


process.Timing = cms.Service("Timing", summaryOnly = cms.untracked.bool(True))


#--- Load config fragment that configures vertex producer
process.load('TMTrackTrigger.l1VertexFinder.VertexProducer_cff')

process.p = cms.Path(process.VertexProducer)
# process.e = cms.EndPath(process.out)
