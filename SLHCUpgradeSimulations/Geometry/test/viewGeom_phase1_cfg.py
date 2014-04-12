import FWCore.ParameterSet.Config as cms

from FWCore.MessageLogger.MessageLogger_cfi import *
process = cms.Process("IGUANA")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("SLHCUpgradeSimulations.Geometry.PhaseI_cmsSimIdealGeometryXML_cfi")
process.load("Configuration.StandardSequences.MagneticField_cff")


process.VisConfigurationService = cms.Service("VisConfigurationService",
                                              Views = cms.untracked.vstring('3D Window'),
                                              ContentProxies = cms.untracked.vstring('Simulation/Core', 
                                                                                     'Simulation/Geometry', 
                                                                                     'Reco/CMS Magnetic Field')
                                              )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )
process.source = cms.Source("EmptySource")

process.prod = cms.EDProducer("GeometryProducer",
                              MagneticField = cms.PSet(
    delta = cms.double(1.0)
    ),
                              UseMagneticField = cms.bool(False),
                              UseSensitiveDetectors = cms.bool(False)
                              )

process.p = cms.Path(process.prod)
