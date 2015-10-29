import FWCore.ParameterSet.Config as cms

process = cms.Process("SimHitStudy")
process.load("SimG4CMS.Calo.HOSimHitStudy_cfi")
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.load("IOMC.EventVertexGenerators.VtxSmearedGauss_cfi")
process.load("Geometry.CMSCommonData.cmsSimIdealGeometryXML_cfi")
process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")
process.load("Geometry.HcalCommonData.hcalParameters_cfi")
process.load("Geometry.HcalCommonData.hcalDDDSimConstants_cfi")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.EventContent.EventContent_cff")
process.load("SimG4Core.Application.g4SimHits_cfi")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['run1_mc']

process.load("IOMC.RandomEngine.IOMC_cff")
process.RandomNumberGeneratorService.generator.initialSeed = 456789
process.RandomNumberGeneratorService.g4SimHits.initialSeed = 9876
process.RandomNumberGeneratorService.VtxSmeared.initialSeed = 123456789

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(50000)
)

process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('cout'),
    categories = cms.untracked.vstring('FwkJob', 'HitStudy'),
    debugModules = cms.untracked.vstring('*'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('DEBUG'),
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        HitStudy = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        FwkJob = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        )
    )
)

#process.Timing = cms.Service("Timing")

process.source = cms.Source("EmptySource",
    firstRun        = cms.untracked.uint32(1),
    firstEvent      = cms.untracked.uint32(1)
)

process.generator = cms.EDProducer("FlatRandomEGunProducer",
    PGunParameters = cms.PSet(
        PartID = cms.vint32(13),
        MinEta = cms.double(-1.305),
        MaxEta = cms.double(1.305),
        MinPhi = cms.double(-3.14159265359),
        MaxPhi = cms.double(3.14159265359),
        MinE   = cms.double(150.),
        MaxE   = cms.double(150.)
    ),
    Verbosity       = cms.untracked.int32(0),
    AddAntiParticle = cms.bool(False)
)

process.TFileService = cms.Service("TFileService",
    fileName = cms.string('simHitStudy_150.root')
)

process.g4SimHits.HCalSD.TestNumberingScheme = True
process.hoSimHitStudy.TestNumbering          = True
process.hoSimHitStudy.PrintExcessEnergy      = False
process.hoSimHitStudy.MaxEnergy = 10.0
process.hoSimHitStudy.ScaleEB   = 1.02
process.hoSimHitStudy.ScaleHB   = 104.4
process.hoSimHitStudy.ScaleHO   = 2.33

process.p1 = cms.Path(process.generator*process.VtxSmeared*process.g4SimHits*process.hoSimHitStudy)

