import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.load("IOMC.EventVertexGenerators.VtxSmearedGauss_cfi")
process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")
process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")
process.load("Geometry.HcalCommonData.hcalParameters_cfi")
process.load("Geometry.HcalCommonData.hcalDDDSimConstants_cfi")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.EventContent.EventContent_cff")
process.load("SimG4Core.Application.g4SimHits_cfi")
process.load("SimG4CMS.Calo.CaloSimHitStudy_cfi")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['run1_mc']

process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('cout'),
    categories = cms.untracked.vstring('HitStudy', 'G4cout', 'G4cerr'),
#    debugModules = cms.untracked.vstring('*'),
    cout = cms.untracked.PSet(
#        threshold = cms.untracked.string('DEBUG'),
        INFO = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        DEBUG = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        G4cerr = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        G4cout = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        HitStudy = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        )
    )
)

process.load("IOMC.RandomEngine.IOMC_cff")
process.RandomNumberGeneratorService.generator.initialSeed = 456789
process.RandomNumberGeneratorService.g4SimHits.initialSeed = 9876
process.RandomNumberGeneratorService.VtxSmeared.initialSeed = 123456789

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(2000)
)

process.source = cms.Source("EmptySource",
    firstRun        = cms.untracked.uint32(1),
    firstEvent      = cms.untracked.uint32(1)
)

process.generator = cms.EDProducer("FlatRandomEGunProducer",
    PGunParameters = cms.PSet(
        PartID = cms.vint32(211),
        MinEta = cms.double(1.2615),
        MaxEta = cms.double(1.2615),
        MinPhi = cms.double(-3.14159265359),
        MaxPhi = cms.double(3.14159265359),
        MinE   = cms.double(50.),
        MaxE   = cms.double(50.)
    ),
    Verbosity       = cms.untracked.int32(0),
    AddAntiParticle = cms.bool(False)
)

process.TFileService = cms.Service("TFileService",
    fileName = cms.string('runScale14_QGSP_BERT_EML.root')
)

process.p1 = cms.Path(process.generator*process.VtxSmeared*process.g4SimHits*process.caloSimHitStudy)
process.caloSimHitStudy.MaxEnergy = 60.0
process.caloSimHitStudy.TimeCut   = 100.0
process.caloSimHitStudy.MIPCut   = 0.75
process.g4SimHits.Physics.type = 'SimG4Core/Physics/QGSP_BERT_EML'

