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

from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['mc']

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


process.Comp = cms.EDAnalyzer("Digi2Raw2Digi",
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

process.outpath = cms.EndPath(process.USER)
