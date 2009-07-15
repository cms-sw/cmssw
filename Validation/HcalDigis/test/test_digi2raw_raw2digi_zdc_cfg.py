import FWCore.ParameterSet.Config as cms

process = cms.Process("VALID")
# import of standard configurations
process.load('Configuration/StandardSequences/Services_cff')
process.load("Configuration.StandardSequences.SimulationRandomNumberGeneratorSeeds_cff")
#process.load("Configuration.StandardSequences.VtxSmearedGauss_cff")
process.load('Configuration/StandardSequences/VtxSmearedEarly10TeVCollision_cff')
process.load("Configuration.StandardSequences.Simulation_cff")
process.load("Configuration.StandardSequences.Digi_cff")
process.load('Configuration/StandardSequences/DigiToRaw_cff')
#process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.MixingNoPileUp_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load('Configuration/StandardSequences/GeometryExtended_cff')
#process.load('Configuration/StandardSequences/GeometryHCAL_cff')
process.load('Configuration/StandardSequences/Generator_cff')
process.load('Configuration/StandardSequences/MagneticField_0T_cff')
process.load('Configuration/StandardSequences/SimExtended_cff')
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load('Configuration/StandardSequences/EndOfProcess_cff')
process.load('Configuration/EventContent/EventContent_cff')
process.GlobalTag.globaltag = 'MC_31X_V1::All'

process.load('Configuration/StandardSequences/DigiToRaw_cff')
process.load('Configuration/StandardSequences/RawToDigi_cff')
process.hcalDigis.InputLabel = 'hcalRawData'

process.load("DQMServices.Core.DQM_cfg")
process.DQM.collectorHost = ''

# The number of events to be processed.
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10))

process.source = cms.Source("EmptySource")
process.generator = cms.EDProducer("FlatRandomEGunProducer",
    PGunParameters = cms.PSet(
        # you can request more than 1 particle
        PartID = cms.vint32(2112),
        MinEta = cms.double(15.0),
        MaxEta = cms.double(18.0),
        MinPhi = cms.double(-3.14159265359),
        MaxPhi = cms.double(3.14159265359),
        MinE   = cms.double(300.0),
        MaxE   = cms.double(300.0)
    ),
    firstRun = cms.untracked.uint32(1),
    AddAntiParticle = cms.bool(False)
)

#process.g4SimHits.UseMagneticField = False
#process.VtxSmeared.SigmaX = 0.00001
#process.VtxSmeared.SigmaY = 0.00001
#process.VtxSmeared.SigmaZ = 0.00001

# try to fix no pcaloHit problem
process.common_maximum_timex = cms.PSet(
    MaxTrackTime  = cms.double(1000.0),
    MaxTimeNames  = cms.vstring('ZDCRegion','CastorRegion'),
    MaxTrackTimes = cms.vdouble(2000.0,0)
)
process.g4SimHits.StackingAction = cms.PSet(
    process.common_heavy_suppression,
    process.common_maximum_timex,
    TrackNeutrino = cms.bool(False),
    KillHeavy     = cms.bool(False),
    SaveFirstLevelSecondary = cms.untracked.bool(True),
    SavePrimaryDecayProductsAndConversionsInTracker = cms.untracked.bool(True),
    SavePrimaryDecayProductsAndConversionsInCalo    = cms.untracked.bool(True),
    SavePrimaryDecayProductsAndConversionsInMuon    = cms.untracked.bool(True)
)
process.g4SimHits.SteppingAction = cms.PSet(
    process.common_maximum_timex,
    KillBeamPipe            = cms.bool(True),
    CriticalEnergyForVacuum = cms.double(2.0),
    CriticalDensity         = cms.double(1e-15),
    EkinNames               = cms.vstring(),
    EkinThresholds          = cms.vdouble(),
    EkinParticles           = cms.vstring(),
    Verbosity               = cms.untracked.int32(0)
)

process.Comp = cms.EDFilter("Digi2Raw2Digi",
    digiLabel1 = cms.InputTag("simHcalDigis"),
    digiLabel2 = cms.InputTag("hcalDigis"),
    outputFile = cms.untracked.string('histo.root')
)

process.USER = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('output.root'),
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('GEN-SIM-RECO'),
        filterName = cms.untracked.string('')
    )
)


#--- this is required for after310pre6 change in g4SimHits input collection
#--- which is by default now HepMCProductLabel = cms.string('LHCTransport')!
#
#process.g4SimHits.Generator.HepMCProductLabel = 'generator'


### process.VtxSmearedCommon.InputTag = "generator"

from SimTransport.HectorProducer.HectorTransportZDC_cfi import *

process.p = cms.Path(
 process.generator * process.VtxSmeared * 
 LHCTransport *
 process.g4SimHits * process.mix *
 process.simHcalUnsuppressedDigis * process.simHcalDigis *
# process.simCastorDigis *
 process.hcalRawData *
 process.hcalDigis  *
 process.Comp
)

