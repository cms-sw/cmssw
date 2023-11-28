import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Eras import eras
process = cms.Process("TEST",eras.Run3)
### RANDOM setting (change last digit(s) to make runs different !)
process.load("Configuration.StandardSequences.SimulationRandomNumberGeneratorSeeds_cff")

process.load('FWCore.MessageService.MessageLogger_cfi')
process.load("SimG4Core.Application.g4SimHits_cfi")
process.load("SimCalorimetry.Configuration.ecalDigiSequence_cff")
process.load("SimCalorimetry.Configuration.hcalDigiSequence_cff")
process.load("SimGeneral.PileupInformation.AddPileupSummary_cfi")
process.load("Configuration.StandardSequences.Reconstruction_cff")

process.load("Configuration.EventContent.EventContent_cff")
process.load("SimGeneral.MixingModule.mixNoPU_cfi")
process.load("Configuration.StandardSequences.DigiToRaw_cff")
process.load("Configuration.StandardSequences.RawToDigi_cff")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = autoCond["phase1_2022_realistic"]

process.load("IOMC.EventVertexGenerators.VtxSmearedGauss_cfi")
process.load("Configuration.StandardSequences.GeometryDB_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.g4SimHits.UseMagneticField = False

from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['phase1_2022_realistic']

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(500) 
)

if 'MessageLogger' in process.__dict__:
    process.MessageLogger.HcalSim=dict()

process.source = cms.Source("EmptySource")
process.generator = cms.EDProducer("FlatRandomEGunProducer",
    PGunParameters = cms.PSet(
        PartID = cms.vint32(13),     #mu ieta=15-18
        MinEta = cms.double(1.22),
        MaxEta = cms.double(1.56),
        MinPhi = cms.double(-3.15926),
        MaxPhi = cms.double(3.15926),
        MinE   = cms.double(20.0),
        MaxE   = cms.double(20.0)
    ),
    firstRun = cms.untracked.uint32(1),
    AddAntiParticle = cms.bool(False)
)

process.VtxSmeared.SigmaX = 0.00001
process.VtxSmeared.SigmaY = 0.00001
process.VtxSmeared.SigmaZ = 0.00001

process.FEVT = cms.OutputModule("PoolOutputModule",
     outputCommands = process.FEVTDEBUGEventContent.outputCommands,
     splitLevel = cms.untracked.int32(0),
     fileName = cms.untracked.string("output.root")
)

process.VtxSmeared.src = cms.InputTag("generator", "unsmeared") 
process.generatorSmeared = cms.EDProducer("GeneratorSmearedProducer")
process.g4SimHits.Generator.HepMCProductLabel = cms.InputTag('VtxSmeared')
process.g4SimHits.LHCTransport = False

process.p = cms.Path(
 process.generator *
 process.VtxSmeared *
 process.generatorSmeared *
 process.g4SimHits *
 process.mix *
 process.ecalDigiSequence * 
 process.hcalDigiSequence *
 process.addPileupInfo *
 process.bunchSpacingProducer *
 process.ecalPacker *
 process.esDigiToRaw *
 process.hcalRawData *
 process.rawDataCollector *
 process.ecalDigis *
 process.ecalPreshowerDigis *
 process.hcalDigis 
)

###process.outpath = cms.EndPath(process.FEVT)
