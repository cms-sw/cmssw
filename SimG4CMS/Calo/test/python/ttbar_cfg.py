import FWCore.ParameterSet.Config as cms

process = cms.Process("Sim")
process.load("SimG4CMS.Calo.PythiaTT_cfi")

process.load("IOMC.EventVertexGenerators.VtxSmearedGauss_cfi")

process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")

process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")

process.load("Configuration.StandardSequences.MagneticField_cff")

process.load("SimG4Core.Application.g4SimHits_cfi")

process.load("SimG4CMS.Calo.CaloSimHitStudy_cfi")

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(50)
)

process.MessageLogger = cms.Service("MessageLogger",
    cout = cms.untracked.PSet(
        noTimeStamps = cms.untracked.bool(False),
        G4cerr = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        PhysicsList = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        MemoryCheck = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        HitStudy = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        FwkJob = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        G4cout = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        )
    ),
    categories = cms.untracked.vstring('FwkJob', 
        'PhysicsList', 
        'G4cout', 
        'G4cerr', 
        'HitStudy', 
        'MemoryCheck'),
    destinations = cms.untracked.vstring('cout')
)

process.Timing = cms.Service("Timing")

process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck",
    oncePerEventMode = cms.untracked.bool(True),
    showMallocInfo = cms.untracked.bool(True),
    dump = cms.untracked.bool(True),
    ignoreTotal = cms.untracked.int32(1)
)

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    moduleSeeds = cms.PSet(
        generator = cms.untracked.uint32(456789),
        g4SimHits = cms.untracked.uint32(9876),
        VtxSmeared = cms.untracked.uint32(98765432)
    ),
    sourceSeed = cms.untracked.uint32(123456789)
)

process.rndmStore = cms.EDProducer("RandomEngineStateProducer")

process.TFileService = cms.Service("TFileService",
    fileName = cms.string('ttbar_QGSP_BERT_EMV.root')
)

# Event output
process.load("Configuration.EventContent.EventContent_cff")

process.o1 = cms.OutputModule("PoolOutputModule",
    process.FEVTSIMEventContent,
    fileName = cms.untracked.string('simevent_ttbar_QGSP_BERT_EMV.root')
)

process.p1 = cms.Path(process.generator*process.VtxSmeared*process.g4SimHits*process.caloSimHitStudy*process.rndmStore)
process.outpath = cms.EndPath(process.o1)
process.generator.pythiaHepMCVerbosity = False
process.generator.pythiaPylistVerbosity = 0
process.g4SimHits.Physics.type = 'SimG4Core/Physics/QGSP_BERT_EMV'
process.g4SimHits.StackingAction.TrackNeutrino = False
process.g4SimHits.Generator.MinPhiCut = -5.5
process.g4SimHits.Generator.MaxPhiCut = 5.5
process.g4SimHits.G4Commands = ['/physics_engine/neutron/energyLimit 1 MeV', '/physics_engine/neutron/timeLimit 0.001 ms']

