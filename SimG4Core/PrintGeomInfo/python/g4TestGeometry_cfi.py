import FWCore.ParameterSet.Config as cms

def checkOverlap(process):

    process.load("SimGeneral.HepPDTESSource.pdt_cfi")

    process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")
    process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")
    process.load("Geometry.EcalCommonData.ecalSimulationParameters_cff")
    process.load("Geometry.HcalCommonData.hcalDDDSimConstants_cff")
    process.load("IOMC.RandomEngine.IOMC_cff")
    process.load('IOMC.EventVertexGenerators.VtxSmearedFlat_cfi')
    process.load('GeneratorInterface.Core.generatorSmeared_cfi')
    process.load('FWCore.MessageService.MessageLogger_cfi')

    process.load("SimG4Core.Application.g4SimHits_cfi")

    process.source = cms.Source("EmptySource")

    process.generator = cms.EDProducer("FlatRandomEGunProducer",
        PGunParameters = cms.PSet(
            PartID = cms.vint32(14),
            MinEta = cms.double(-3.5),
            MaxEta = cms.double(3.5),
            MinPhi = cms.double(-3.14159265359),
            MaxPhi = cms.double(3.14159265359),
            MinE   = cms.double(9.99),
            MaxE   = cms.double(10.01)
        ),
        AddAntiParticle = cms.bool(False),
        Verbosity       = cms.untracked.int32(0),
        firstRun        = cms.untracked.uint32(1)
    )

    process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(1)
    )

    process.p1 = cms.Path(process.generator*process.VtxSmeared*process.generatorSmeared*process.g4SimHits)

    process.g4SimHits.UseMagneticField = False
    process.g4SimHits.Physics.type = 'SimG4Core/Physics/DummyPhysics'
    process.g4SimHits.Physics.DummyEMPhysics = True
    process.g4SimHits.LHCTransport = False

    return(process)
