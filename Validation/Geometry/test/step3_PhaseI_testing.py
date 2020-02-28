# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: step3 --conditions auto:phase1_2017_realistic --pileup_input das:/RelValMinBias_13/CMSSW_8_1_0_pre7-81X_upgrade2017_realistic_v3_UPG17newGT-v1/GEN-SIM -n 10 --era Run2_2017 --eventcontent RECOSIM,DQM -s RAW2DIGI,L1Reco,RECO:reconstruction_trackingOnly,VALIDATION:@trackingOnlyValidation,DQM:@trackingOnlyDQM --datatier GEN-SIM-RECO,DQMIO --pileup AVE_35_BX_25ns --geometry DB:Extended --conditions 81X_upgrade2017_realistic_v3 --no_exec --filein file:step2.root --fileout file:step3.root --nThreads 4 --python_filename fuffa.py
import FWCore.ParameterSet.Config as cms


from Configuration.Eras.Era_Run2_2017_cff import Run2_2017
process = cms.Process('RECO',Run2_2017)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mix_POISSON_average_cfi')
#process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.Geometry.GeometryExtended2017Reco_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Configuration.StandardSequences.L1Reco_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('Configuration.StandardSequences.Validation_cff')
process.load('DQMOffline.Configuration.DQMOfflineMC_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(500)
)

# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
'/store/relval/CMSSW_8_1_0_pre7/RelValTTbar_13/GEN-SIM-DIGI-RAW/PU25ns_81X_upgrade2017_realistic_v3_UPG17PU35newGTresub-v1/10000/1878BE3F-8838-E611-9279-0025905A60F4.root',
'/store/relval/CMSSW_8_1_0_pre7/RelValTTbar_13/GEN-SIM-DIGI-RAW/PU25ns_81X_upgrade2017_realistic_v3_UPG17PU35newGTresub-v1/10000/1A74F012-8A38-E611-AFD0-0025905A6070.root',
'/store/relval/CMSSW_8_1_0_pre7/RelValTTbar_13/GEN-SIM-DIGI-RAW/PU25ns_81X_upgrade2017_realistic_v3_UPG17PU35newGTresub-v1/10000/3059622D-8A38-E611-B212-0025905A6056.root',
'/store/relval/CMSSW_8_1_0_pre7/RelValTTbar_13/GEN-SIM-DIGI-RAW/PU25ns_81X_upgrade2017_realistic_v3_UPG17PU35newGTresub-v1/10000/3E7BBA13-8A38-E611-8899-0025905A48C0.root'
    ),
    secondaryFileNames = cms.untracked.vstring()
)

process.options = cms.untracked.PSet(

)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('step3 nevts:10'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# Output definition

process.RECOSIMoutput = cms.OutputModule("PoolOutputModule",
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('GEN-SIM-RECO'),
        filterName = cms.untracked.string('')
    ),
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
    fileName = cms.untracked.string('file:matbdgForReco_FromReco_TTbarPhaseI.root'),
    outputCommands = process.RECOSIMEventContent.outputCommands,
    splitLevel = cms.untracked.int32(0)
)

process.DQMoutput = cms.OutputModule("DQMRootOutputModule",
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('DQMIO'),
        filterName = cms.untracked.string('')
    ),
    fileName = cms.untracked.string('file:matbdgForReco_FromReco_TTbarPhaseI_inDQM.root'),
    outputCommands = process.DQMEventContent.outputCommands,
    splitLevel = cms.untracked.int32(0)
)

# Additional output definition

# Other statements
process.mix.input.nbPileupEvents.averageNumber = cms.double(35.000000)
process.mix.bunchspace = cms.int32(25)
process.mix.minBunch = cms.int32(-12)
process.mix.maxBunch = cms.int32(3)
process.mix.input.fileNames = cms.untracked.vstring(['/store/relval/CMSSW_8_1_0_pre7/RelValMinBias_13/GEN-SIM/81X_upgrade2017_realistic_v3_UPG17newGT-v1/10000/441C0D82-6937-E611-BDA5-0025905B85C0.root', '/store/relval/CMSSW_8_1_0_pre7/RelValMinBias_13/GEN-SIM/81X_upgrade2017_realistic_v3_UPG17newGT-v1/10000/5E259E37-6E37-E611-BAE4-0025905B85FC.root', '/store/relval/CMSSW_8_1_0_pre7/RelValMinBias_13/GEN-SIM/81X_upgrade2017_realistic_v3_UPG17newGT-v1/10000/6CC91B80-6937-E611-AA2B-0025905A6080.root', '/store/relval/CMSSW_8_1_0_pre7/RelValMinBias_13/GEN-SIM/81X_upgrade2017_realistic_v3_UPG17newGT-v1/10000/78082F77-7837-E611-9BEB-0CC47A78A30E.root', '/store/relval/CMSSW_8_1_0_pre7/RelValMinBias_13/GEN-SIM/81X_upgrade2017_realistic_v3_UPG17newGT-v1/10000/7C751DEC-6C37-E611-9AF5-0025905A60CE.root', '/store/relval/CMSSW_8_1_0_pre7/RelValMinBias_13/GEN-SIM/81X_upgrade2017_realistic_v3_UPG17newGT-v1/10000/A2808D84-6F37-E611-BBE1-0CC47A745294.root', '/store/relval/CMSSW_8_1_0_pre7/RelValMinBias_13/GEN-SIM/81X_upgrade2017_realistic_v3_UPG17newGT-v1/10000/B22CF4B2-7837-E611-8449-0025905A6084.root', '/store/relval/CMSSW_8_1_0_pre7/RelValMinBias_13/GEN-SIM/81X_upgrade2017_realistic_v3_UPG17newGT-v1/10000/D4A12288-6F37-E611-A354-0025905B85D8.root', '/store/relval/CMSSW_8_1_0_pre7/RelValMinBias_13/GEN-SIM/81X_upgrade2017_realistic_v3_UPG17newGT-v1/10000/F0E482B2-6C37-E611-B9AD-0CC47A4D760A.root'])
process.mix.playback = True
process.mix.digitizers = cms.PSet()
for a in process.aliases: delattr(process, a)
process.RandomNumberGeneratorService.restoreStateLabel=cms.untracked.string("randomEngineStateProducer")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, '81X_upgrade2017_realistic_v3', '')


from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
process.materialdumper = DQMEDAnalyzer('TrackingRecoMaterialAnalyser',
        tracks = cms.InputTag("generalTracks"),
        vertices = cms.InputTag("offlinePrimaryVertices"),
        DoPredictionsOnly = cms.bool(False),
        Fitter = cms.string('KFFitterForRefitInsideOut'),
        TrackerRecHitBuilder = cms.string('WithAngleAndTemplate'),
        Smoother = cms.string('KFSmootherForRefitInsideOut'),
        MuonRecHitBuilder = cms.string('MuonRecHitBuilder'),
        RefitDirection = cms.string('alongMomentum'),
        RefitRPCHits = cms.bool(True),
        Propagator = cms.string('SmartPropagatorAnyRKOpposite'),
        #Propagators
        PropagatorAlong = cms.string("RungeKuttaTrackerPropagator"),
        PropagatorOpposite = cms.string("RungeKuttaTrackerPropagatorOpposite")
)




# Path and EndPath definitions
process.raw2digi_step = cms.Path(process.RawToDigi)
process.L1Reco_step = cms.Path(process.L1Reco)
process.reconstruction_step = cms.Path(process.reconstruction_trackingOnly)
process.prevalidation_step = cms.Path(process.globalPrevalidationTrackingOnly)
process.dqmoffline_step = cms.Path(process.DQMOfflineTracking)
process.dqmofflineOnPAT_step = cms.Path(process.PostDQMOffline)
process.validation_step = cms.EndPath(process.globalValidationTrackingOnly)
process.RECOSIMoutput_step = cms.EndPath(process.RECOSIMoutput)
process.DQMoutput_step = cms.EndPath(process.DQMoutput)
process.materialdumper_step = cms.Path(process.materialdumper)

# Schedule definition
process.schedule = cms.Schedule(process.raw2digi_step,process.L1Reco_step,process.reconstruction_step,process.materialdumper_step,process.prevalidation_step,process.validation_step,process.dqmoffline_step,process.dqmofflineOnPAT_step,process.RECOSIMoutput_step,process.DQMoutput_step)

#Setup FWK for multithreaded
process.options.numberOfThreads=cms.untracked.uint32(8)
process.options.numberOfStreams=cms.untracked.uint32(0)

# customisation of the process.

# Automatic addition of the customisation function from SimGeneral.MixingModule.fullMixCustomize_cff
from SimGeneral.MixingModule.fullMixCustomize_cff import setCrossingFrameOn 

#call to customisation function setCrossingFrameOn imported from SimGeneral.MixingModule.fullMixCustomize_cff
process = setCrossingFrameOn(process)

# End of customisation functions

