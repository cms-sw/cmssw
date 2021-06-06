import FWCore.ParameterSet.Config as cms

#------------------------------------------------------------
# This is a central part of geometry test implemented in this 
# sub-library. This file is used by test fragments from tets
# sub-directory. It is mandatory for these tests disabling of 
# static build of Simulation:
#
# scram b disable-biglib 
#
#------------------------------------------------------------

process = cms.Process("CheckOverlap")

process.load("SimGeneral.HepPDTESSource.pdt_cfi")

#process.load("Geometry.CMSCommonData.cmsExtendedGeometry2015XML_cfi")
process.load("Geometry.CMSCommonData.cmsExtendedGeometry2015devXML_cfi")

process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")
process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")
process.load("Geometry.EcalCommonData.ecalSimulationParameters_cff")
process.load("Geometry.HcalCommonData.hcalDDDSimConstants_cff")

process.load("SimG4Core.Application.g4SimHits_cfi")

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        G4cerr = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        G4cout = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        enable = cms.untracked.bool(True),
        threshold = cms.untracked.string('DEBUG')
    ),
    debugModules = cms.untracked.vstring('*')
)

process.load("IOMC.RandomEngine.IOMC_cff")

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

process.p1 = cms.Path(process.generator*process.g4SimHits)

process.g4SimHits.UseMagneticField = False
process.g4SimHits.Physics.type = 'SimG4Core/Physics/DummyPhysics'
process.g4SimHits.Physics.DummyEMPhysics = True


