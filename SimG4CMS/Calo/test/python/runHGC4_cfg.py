import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Era_Phase2C11_cff import Phase2C11

process = cms.Process("PROD",Phase2C11)
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.load("IOMC.EventVertexGenerators.VtxSmearedGauss_cfi")
process.load("Configuration.Geometry.GeometryExtended2026D77Reco_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.EventContent.EventContent_cff")
process.load('Configuration.StandardSequences.Generator_cff')
process.load('Configuration.StandardSequences.SimIdeal_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('SimG4CMS.Calo.hgcalHitIdCheck_cff')
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic_T21', '')

if hasattr(process,'MessageLogger'):
    process.MessageLogger.HGCalGeom=dict()
    process.MessageLogger.HGCSim=dict()
    process.MessageLogger.CaloSim=dict()
    process.MessageLogger.HitStudy=dict()
#   process.MessageLogger.SimG4CoreGeometry=dict()

process.load("IOMC.RandomEngine.IOMC_cff")
process.RandomNumberGeneratorService.generator.initialSeed = 456789
process.RandomNumberGeneratorService.g4SimHits.initialSeed = 9876
process.RandomNumberGeneratorService.VtxSmeared.initialSeed = 123456789

process.Timing = cms.Service("Timing")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(5)
)

process.source = cms.Source("EmptySource",
    firstRun        = cms.untracked.uint32(1),
    firstEvent      = cms.untracked.uint32(1)
)

process.generator = cms.EDProducer("FlatRandomEGunProducer",
    PGunParameters = cms.PSet(
        PartID = cms.vint32(211),
#       MinEta = cms.double(2.95),
#       MaxEta = cms.double(3.01),
#       MinEta = cms.double(1.69),
#       MaxEta = cms.double(2.32),
        MinEta = cms.double(1.50),
        MaxEta = cms.double(2.00),
        MinPhi = cms.double(-3.1415926),
        MaxPhi = cms.double(-1.5707963),
        MinE   = cms.double(100.00),
        MaxE   = cms.double(100.00)
    ),
    Verbosity       = cms.untracked.int32(0),
    AddAntiParticle = cms.bool(True)
)

process.output = cms.OutputModule("PoolOutputModule",
    process.FEVTSIMEventContent,
    fileName = cms.untracked.string('hgcV14.root')
)

process.generation_step = cms.Path(process.pgen)
process.simulation_step = cms.Path(process.psim)
process.analysis_step = cms.Path(process.hgcalHitIdCheckEE+process.hgcalHitIdCheckHEF+process.hgcalHitIdCheckHEB)
process.out_step = cms.EndPath(process.output)

process.g4SimHits.Physics.type = 'SimG4Core/Physics/FTFP_BERT_EMM'
process.g4SimHits.Physics.DefaultCutValue          = 0.1
process.g4SimHits.HGCSD.CornerMinMask              = 3
process.g4SimHits.CaloSD.UseFineCalo               = [2]
process.g4SimHits.CaloTrkProcessing.UseFineCalo    = [2]
process.g4SimHits.TrackingAction.UseFineCalo       = [2]
process.g4SimHits.CaloSD.DoFineCalo                = True
process.g4SimHits.CaloTrkProcessing.DoFineCalo     = True
process.g4SimHits.TrackingAction.DoFineCalo        = True
process.g4SimHits.CaloSD.EminFineTrack             = 1000.0
process.g4SimHits.CaloTrkProcessing.EminFineTrack  = 1000.0
process.g4SimHits.TrackingAction.EminFineTrack     = 1000.0
process.hgcalHitIdCheckHEB.Verbosity               = 1

# Schedule definition
process.schedule = cms.Schedule(process.generation_step,
                                process.simulation_step,
                                process.analysis_step,
                                process.out_step
                                )

# filter all path with the production filter sequence
for path in process.paths:
        getattr(process,path)._seq = process.generator * getattr(process,path)._seq
