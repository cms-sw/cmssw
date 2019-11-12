# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: step2 --conditions auto:phase2_realistic -s DIGI:pdigi_valid,L1,L1TrackTrigger,DIGI2RAW,HLT:@fake2,RAW2DIGI,L1Reco,RECO --datatier GEN-SIM-RECO -n 10 --geometry Extended2023D41 --era Phase2 --eventcontent FEVTDEBUGHLT --filein file:SingleMuPt1000_pythia8_cfi_GEN_SIM.root --runUnscheduled --no_exec
import FWCore.ParameterSet.Config as cms


from Configuration.Eras.Era_Phase2C8_cff import Phase2C8
process = cms.Process('Phase2PixelNtuple',Phase2C8)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
#process.load('SimGeneral.MixingModule.mix_POISSON_average_cfi')
process.load('Configuration.Geometry.GeometryExtended2026D41Reco_cff')
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
       '/store/relval/CMSSW_10_6_0_patch2/RelValSingleMuPt10/GEN-SIM/106X_upgrade2023_realistic_v3_2023D41noPU-v1/10000/7377ED92-245C-CC4D-9F05-25ABB5522A08.root'
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
process.ReadLocalMeasurement = cms.EDAnalyzer("Phase2PixelNtuple",
   trackProducer = cms.InputTag("generalTracks"),
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
   usePhase2Tracker = cms.bool(True),
   pixelSimLinkSrc = cms.InputTag("simSiPixelDigis", "Pixel"),
   phase2TrackerSimLinkSrc = cms.InputTag("simSiPixelDigis", "Tracker")
)


# Additional output definition

# Other statements
process.mix.digitizers = cms.PSet(process.theDigitizersValid)

# This pset is specific for producing simulated events for the designers of the PROC (InnerTracker)
# They need pixel RecHits where the charge is stored with high-granularity and large dinamic range

# digitizer
process.mix.digitizers.pixel.PixelDigitizerAlgorithm.AdcFullScale   = cms.int32(255)
process.mix.digitizers.pixel.PixelDigitizerAlgorithm.ElectronPerAdc = cms.double(135.)
process.mix.digitizers.pixel.PixelDigitizerAlgorithm.AddXTalk = cms.bool(False)

# clusterizer
process.siPixelClusters.ElectronPerADCGain  = cms.double(135.)
process.siPixelClustersPreSplitting.ElectronPerADCGain  = cms.double(135.)

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic_T14', '')

# Path and EndPath definitions
process.digitisation_step = cms.Path(process.pdigi_valid)
process.L1simulation_step = cms.Path(process.SimL1Emulator)
process.L1TrackTrigger_step = cms.Path(process.L1TrackTrigger)
process.digi2raw_step = cms.Path(process.DigiToRaw)
process.raw2digi_step = cms.Path(process.RawToDigi)
process.L1Reco_step = cms.Path(process.L1Reco)
process.reconstruction_step = cms.Path(process.reconstruction)
process.user_step = cms.Path(process.ReadLocalMeasurement + process.mcverticesanalyzer)
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
#do not add changes to your config after this point (unless you know what you are doing)
from FWCore.ParameterSet.Utilities import convertToUnscheduled
process=convertToUnscheduled(process)


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

