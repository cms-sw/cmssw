import FWCore.ParameterSet.Config as cms

process = cms.Process("HFShowerLib")

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.load("IOMC.EventVertexGenerators.VtxSmearedGauss_cfi")
process.load("Geometry.CMSCommonData.cmsExtendedGeometryHFLibraryXML_cfi")
process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cff")
process.load("Geometry.EcalCommonData.ecalSimulationParameters_cff")
process.load("Geometry.HcalCommonData.hcalDDDSimConstants_cff")
process.load("Geometry.MuonNumbering.muonGeometryConstants_cff")
process.load("Geometry.MuonNumbering.muonOffsetESProducer_cff")

process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.EventContent.EventContent_cff")
process.load('Configuration.StandardSequences.Generator_cff')
process.load('Configuration.StandardSequences.SimIdeal_cff')

process.load("Configuration.StandardSequences.Services_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.g4SimHits.UseMagneticField = False

process.load("Configuration.StandardSequences.Services_cff")
process.RandomNumberGeneratorService.generator.initialSeed = 12345

from IOMC.RandomEngine.RandomServiceHelper import RandomNumberServiceHelper
randSvc = RandomNumberServiceHelper(process.RandomNumberGeneratorService)
randSvc.populate()

process.load('FWCore.MessageService.MessageLogger_cfi')
if hasattr(process,'MessageLogger'):
    process.MessageLogger.FiberSim=dict()
    process.MessageLogger.HcalSim=dict()
    process.MessageLogger.HFShower=dict()
    process.MessageLogger.HcalForwardLib=dict()

process.Timing = cms.Service("Timing")

process.source = cms.Source("EmptySource")

process.generator = cms.EDProducer("FlatRandomEGunProducer",
    PGunParameters = cms.PSet(
        PartID = cms.vint32(11),
        MinEta = cms.double(4),
        MaxEta = cms.double(4),
        MinPhi = cms.double(0),
        MaxPhi = cms.double(0),
        MinE     = cms.double(10),
        MaxE     = cms.double(10)
    ),
    Verbosity = cms.untracked.int32(0),
    AddAntiParticle = cms.bool(False),
    firstRun = cms.untracked.uint32(1)
)

process.TFileService = cms.Service("TFileService",
    fileName = cms.string('hfShowerLibSimu.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

# --- make IP(0,0,0) 
process.VtxSmeared.SigmaX = 0.0
process.VtxSmeared.SigmaY = 0.0
process.VtxSmeared.SigmaZ = 0.0

# for GEN produced since 760pre6, for older GEN - just "":
process.VtxSmeared.src = cms.InputTag("generator", "unsmeared")
process.generatorSmeared = cms.EDProducer("GeneratorSmearedProducer")
process.g4SimHits.Generator.HepMCProductLabel = cms.InputTag('VtxSmeared')

process.p1 = cms.Path(
 process.generator *
 process.VtxSmeared *
 process.generatorSmeared *
 process.g4SimHits
)


#process.p1 = cms.Path(cms.SequencePlaceholder("randomEngineStateProducer")+process.VtxSmeared*process.g4SimHits)
#process.p1 = cms.Path(process.generator*process.g4SimHits)
#process.outpath = cms.EndPath(process.o1)
process.g4SimHits.NonBeamEvent = True
process.g4SimHits.Generator.ApplyPCuts   = False
process.g4SimHits.Generator.ApplyEtaCuts = False
#process.g4SimHits.Physics.type = 'SimG4Core/Physics/FTFP_BERT_EMM'
process.g4SimHits.Physics.DefaultCutValue = 0.1
process.g4SimHits.HFShower.ApplyFiducialCut = cms.bool(False)


