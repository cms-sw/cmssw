# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: step3 --conditions auto:phase2_realistic_T21 -s RAW2DIGI,L1Reco,RECO,RECOSIM,PAT,VALIDATION:@phase2Validation+@miniAODValidation,DQM:@phase2+@miniAODDQM --datatier GEN-SIM-RECO,MINIAODSIM,DQMIO -n 10 --geometry Extended2026D66 --era Phase2C11 --eventcontent FEVTDEBUGHLT,MINIAODSIM,DQM --filein file:step2.root --fileout file:step3.root
import FWCore.ParameterSet.Config as cms


from Configuration.Eras.Era_Phase2C11_cff import Phase2C11
process = cms.Process('USER',Phase2C11)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.Geometry.GeometryExtended2026D66Reco_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Configuration.StandardSequences.L1Reco_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('Configuration.StandardSequences.RecoSim_cff')
process.load('PhysicsTools.PatAlgos.slimming.metFilterPaths_cff')
process.load('Configuration.StandardSequences.PATMC_cff')
process.load('Configuration.StandardSequences.Validation_cff')
process.load('DQMServices.Core.DQMStoreNonLegacy_cff')
process.load('DQMOffline.Configuration.DQMOfflineMC_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'file:step2.root'
    )
)


process.options = cms.untracked.PSet(

)

#process.MessageLogger = cms.Service(
#    "MessageLogger",
#    destinations = cms.untracked.vstring(
#        'detailedInfo',
#         ),
#    detailedInfo = cms.untracked.PSet(
#        threshold = cms.untracked.string('DEBUG')
#         ),
#    debugModules = cms.untracked.vstring(
#        'reconstruction_step',
#        )
#    )
#

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('step2 nevts:10'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# MC vertice analyzer
process.load("Validation.RecoVertex.mcverticesanalyzer_cfi")
process.mcverticesanalyzer.pileupSummaryCollection = cms.InputTag("addPileupInfo","","HLT")

# # # -- Trajectory producer
process.load("RecoTracker.TrackProducer.TrackRefitters_cff")
process.TrackRefitter.src = 'generalTracks'
process.TrackRefitter.NavigationSchool = ""

process.ReadLocalMeasurement = cms.EDAnalyzer("Phase2PixelNtuple",
                                              trackProducer = cms.InputTag("generalTracks"),
                                              trajectoryInput = cms.InputTag('TrackRefitter::USER'),
                                              #verbose = cms.untracked.bool(True),
                                              #picky = cms.untracked.bool(False),                                             
                                              ### for using track hit association
                                              associatePixel = cms.bool(True),
                                              associateStrip = cms.bool(False),
                                              associateRecoTracks = cms.bool(False),
                                              ROUList = cms.vstring('TrackerHitsPixelBarrelLowTof',
                                                                    'TrackerHitsPixelBarrelHighTof',
                                                                    'TrackerHitsPixelEndcapLowTof',
                                                                    'TrackerHitsPixelEndcapHighTof'),
                                              ttrhBuilder = cms.string("WithTrackAngle"),
                                              usePhase2Tracker = cms.bool(True),
                                              pixelSimLinkSrc = cms.InputTag("simSiPixelDigis", "Pixel"),
                                              phase2TrackerSimLinkSrc = cms.InputTag("simSiPixelDigis", "Tracker")
                                          )


# Additional output definition

# Other statements
process.mix.playback = True
process.mix.digitizers = cms.PSet()
for a in process.aliases: delattr(process, a)
process.RandomNumberGeneratorService.restoreStateLabel=cms.untracked.string("randomEngineStateProducer")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic_T21', '')

# Uncomment to use CPE generic instead of template in final fit
#process.load("RecoTracker.TransientTrackingRecHit.TTRHBuilderWithTemplate_cfi")
#process.TTRHBuilderAngleAndTemplate.PixelCPE = cms.string("PixelCPEGeneric")

# Uncomment CPE Template for every step (including seeding)
#process.load("RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi")
#process.siPixelRecHits.CPE = cms.string('PixelCPETemplateReco')
#process.load("RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilder_cfi")
#process.ttrhbwr.PixelCPE = cms.string('PixelCPETemplateReco')

# Path and EndPath definitions
process.raw2digi_step = cms.Path(process.RawToDigi)
process.L1Reco_step = cms.Path(process.L1Reco)
process.reconstruction_step = cms.Path(process.reconstruction)
process.user_step = cms.Path(process.TrackRefitter * process.ReadLocalMeasurement * process.mcverticesanalyzer)
process.endjob_step = cms.EndPath(process.endOfProcess)

# Schedule definition
process.schedule = cms.Schedule(process.raw2digi_step,process.L1Reco_step,process.reconstruction_step,process.user_step,process.endjob_step)
from PhysicsTools.PatAlgos.tools.helpers import associatePatAlgosToolsTask
associatePatAlgosToolsTask(process)

# customisation of the process.

# Automatic addition of the customisation function from SimGeneral.MixingModule.fullMixCustomize_cff
from SimGeneral.MixingModule.fullMixCustomize_cff import setCrossingFrameOn 

#call to customisation function setCrossingFrameOn imported from SimGeneral.MixingModule.fullMixCustomize_cff
process = setCrossingFrameOn(process)

# End of customisation functions

# customisation of the process.

# Automatic addition of the customisation function from PhysicsTools.PatAlgos.slimming.miniAOD_tools
from PhysicsTools.PatAlgos.slimming.miniAOD_tools import miniAOD_customizeAllMC 

#call to customisation function miniAOD_customizeAllMC imported from PhysicsTools.PatAlgos.slimming.miniAOD_tools
process = miniAOD_customizeAllMC(process)

# End of customisation functions

# Customisation from command line

#Have logErrorHarvester wait for the same EDProducers to finish as those providing data for the OutputModule
from FWCore.Modules.logErrorHarvester_cff import customiseLogErrorHarvesterUsingOutputCommands
process = customiseLogErrorHarvesterUsingOutputCommands(process)

# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)

# End adding early deletion
process.TFileService = cms.Service('TFileService', fileName = cms.string("pixelntuple.root")
)

