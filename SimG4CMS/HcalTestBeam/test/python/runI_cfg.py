import FWCore.ParameterSet.Config as cms

process = cms.Process("GEOM")
process.load("SimG4CMS.HcalTestBeam.TB2004GeometryXML_cfi")

process.VisConfigurationService = cms.Service("VisConfigurationService",
    Views = cms.untracked.vstring('3D Window'),
    ContentProxies = cms.untracked.vstring('Simulation/Core', 
        'Simulation/Geometry', 
        'Simulation/MagField')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(0)
)

process.source = cms.Source("EmptySource",
    firstRun   = cms.untracked.uint32(1),
    firstEvent = cms.untracked.uint32(1)
)

process.m = cms.EDProducer("GeometryProducer",
    MagneticField = cms.PSet(
        delta = cms.double(1.0)
    ),
    UseMagneticField = cms.bool(False),
    UseSensitiveDetectors = cms.bool(False)
)

process.p1 = cms.Path(process.m)

