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
    cerr = dict(
        enable = False
    ),
    cout = dict(
        G4cerr = dict(
            limit = -1
        ),
        G4cout = dict(
            limit = -1
        ),
        default = dict(
            limit = 0
        ),
        enable = True,
        threshold = 'DEBUG'
    ),
    debugModules = cms.untracked.vstring('*')
)

process.load("IOMC.RandomEngine.IOMC_cff")

process.source = cms.Source("EmptySource")

process.generator = cms.EDProducer("FlatRandomEGunProducer",
    PGunParameters = dict(
        PartID = 14,
        MinEta = -3.5,
        MaxEta = 3.5,
        MinPhi = -3.14159265359,
        MaxPhi = 3.14159265359,
        MinE   = 9.99,
        MaxE   = 10.01
    ),
    AddAntiParticle = cms.bool(False),
    Verbosity       = cms.untracked.int32(0),
    firstRun        = cms.untracked.uint32(1)
)

process.maxEvents = dict(
    input = 1
)

process.p1 = cms.Path(process.generator*process.g4SimHits)

process.g4SimHits.UseMagneticField = False
process.g4SimHits.Physics.type = 'SimG4Core/Physics/DummyPhysics'
process.g4SimHits.Physics.DummyEMPhysics = True


