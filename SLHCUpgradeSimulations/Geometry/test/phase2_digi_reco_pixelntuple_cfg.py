import FWCore.ParameterSet.Config as cms


from Configuration.Eras.Era_Phase2C9_cff import Phase2C9
process = cms.Process('USER',Phase2C9)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
#process.load('SimGeneral.MixingModule.mix_POISSON_average_cfi')
process.load('Configuration.Geometry.GeometryExtended2026D49Reco_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.Digi_cff')
process.load('Configuration.StandardSequences.SimL1Emulator_cff')
process.load('Configuration.StandardSequences.L1TrackTrigger_cff')
process.load('Configuration.StandardSequences.DigiToRaw_cff')
process.load('HLTrigger.Configuration.HLT_Fake2_cff')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Configuration.StandardSequences.L1Reco_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_11_2_0_pre8/RelValSingleMuPt10/GEN-SIM-RECO/112X_mcRun4_realistic_v3_2026D49noPU-v1/00000/007d817e-9c59-4dec-959b-0f227942cdf0.root'
    )
)


process.options = cms.untracked.PSet(

)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('step2 nevts:10'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# MC vertice analyzer
process.load("Validation.RecoVertex.mcverticesanalyzer_cfi")
process.mcverticesanalyzer.pileupSummaryCollection = cms.InputTag("addPileupInfo","","HLT")

# Output definition

#process.FEVTDEBUGHLToutput = cms.OutputModule("PoolOutputModule",
#    dataset = cms.untracked.PSet(
#        dataTier = cms.untracked.string('GEN-SIM-RECO'),
#        filterName = cms.untracked.string('')
#    ),
#    fileName = cms.untracked.string('step2_DIGI_L1_L1TrackTrigger_DIGI2RAW_HLT_RAW2DIGI_L1Reco_RECO.root'),
#    outputCommands = process.FEVTDEBUGHLTEventContent.outputCommands,
#    splitLevel = cms.untracked.int32(0)
#)

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
process.mix.digitizers = cms.PSet(process.theDigitizersValid)

# This pset is specific for producing simulated events for the designers of the PROC (InnerTracker)
# They need pixel RecHits where the charge is stored with high-granularity and large dinamic range

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic_T15', '')

# Path and EndPath definitions
process.digitisation_step = cms.Path(process.pdigi_valid)
process.L1simulation_step = cms.Path(process.SimL1Emulator)
process.L1TrackTrigger_step = cms.Path(process.L1TrackTrigger)
process.digi2raw_step = cms.Path(process.DigiToRaw)
process.raw2digi_step = cms.Path(process.RawToDigi)
process.L1Reco_step = cms.Path(process.L1Reco)
process.reconstruction_step = cms.Path(process.reconstruction)
process.user_step = cms.Path(process.TrackRefitter * process.ReadLocalMeasurement * process.mcverticesanalyzer)
process.endjob_step = cms.EndPath(process.endOfProcess)
#process.FEVTDEBUGHLToutput_step = cms.EndPath(process.FEVTDEBUGHLToutput)

# Schedule definition
process.schedule = cms.Schedule(process.digitisation_step,process.L1simulation_step,process.L1TrackTrigger_step,process.digi2raw_step)
process.schedule.extend(process.HLTSchedule)
process.schedule.extend([process.raw2digi_step,process.L1Reco_step,process.reconstruction_step,process.user_step,process.endjob_step])
from PhysicsTools.PatAlgos.tools.helpers import associatePatAlgosToolsTask
associatePatAlgosToolsTask(process)

# customisation of the process.

# Automatic addition of the customisation function from HLTrigger.Configuration.customizeHLTforMC
from HLTrigger.Configuration.customizeHLTforMC import customizeHLTforMC 

#call to customisation function customizeHLTforMC imported from HLTrigger.Configuration.customizeHLTforMC
process = customizeHLTforMC(process)

# End of customisation functions

# Customisation from command line

#Have logErrorHarvester wait for the same EDProducers to finish as those providing data for the OutputModule
from FWCore.Modules.logErrorHarvester_cff import customiseLogErrorHarvesterUsingOutputCommands
process = customiseLogErrorHarvesterUsingOutputCommands(process)

# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)
# End adding early deletion
process.TFileService = cms.Service('TFileService',
fileName = cms.string("pixelntuple.root")
)

