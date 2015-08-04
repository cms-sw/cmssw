import FWCore.ParameterSet.Config as cms

process = cms.Process("testHF")

process.load("Configuration.StandardSequences.SimulationRandomNumberGeneratorSeeds_cff")
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

process.load("IOMC.EventVertexGenerators.VtxSmearedGauss_cfi")

#--- Magnetic Field 		
process.load("Configuration.StandardSequences.MagneticField_cff")

#--- Full geometry or only ECAL+HCAL Geometry
process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")
process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")
#process.load("Geometry/CMSCommonData/data/ecalhcalGeometryXML.cfi")

process.load("SimG4Core.Application.g4SimHits_cfi")

process.MessageLogger = cms.Service("MessageLogger")

process.Timing = cms.Service("Timing")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

process.source = cms.Source("EmptySource")

process.generator = cms.EDProducer("FlatRandomEGunProducer",
    VertexSmearing = cms.PSet(refToPSet_ = cms.string("VertexSmearingParameters")),
    PGunParameters = cms.PSet(
        PartID = cms.vint32(211),
        MinEta = cms.double(3.02),
        MaxEta = cms.double(5.02),
        MinPhi = cms.double(-3.14159265359),
        MaxPhi = cms.double(3.14159265359),
        MinE   = cms.double(100.0),
        MaxE   = cms.double(100.0)
    ),
    AddAntiParticle = cms.bool(False),
    Verbosity       = cms.untracked.int32(0),
    firstRun        = cms.untracked.uint32(1)
)

from IOMC.EventVertexGenerators.VtxSmearedGauss_cfi import VertexSmearingParameters
process.generator.VertexSmearing = cms.PSet(
    VertexSmearingParameters
)
process.generator.VertexSmearing.SigmaX = 0.00001
process.generator.VertexSmearing.SigmaY = 0.00001
process.generator.VertexSmearing.SigmaZ = 0.00001

process.mix = cms.EDProducer("MixingModule",
    bunchspace = cms.int32(25)
)

process.o1 = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('my_output.root')
)

process.p1 = cms.Path(process.generator*process.g4SimHits)
process.outpath = cms.EndPath(process.o1)
process.g4SimHits.UseMagneticField = False


