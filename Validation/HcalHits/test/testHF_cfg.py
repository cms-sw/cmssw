import FWCore.ParameterSet.Config as cms

process = cms.Process("testHF")
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

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    moduleSeeds = cms.PSet(
        g4SimHits = cms.untracked.uint32(9876),
        VtxSmeared = cms.untracked.uint32(123456789)
    ),
    sourceSeed = cms.untracked.uint32(135799753)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

process.source = cms.Source("FlatRandomEGunSource",
    PGunParameters = cms.untracked.PSet(
        # you can request more than 1 particle
        PartID = cms.untracked.vint32(211),
        MinEta = cms.untracked.double(3.02),
        MaxEta = cms.untracked.double(5.02),
        MinPhi = cms.untracked.double(-3.14159265359),
        MaxPhi = cms.untracked.double(3.14159265359),
        MinE = cms.untracked.double(100.0),
        MaxE = cms.untracked.double(100.0)
    ),
    firstRun = cms.untracked.uint32(1)
)

process.mix = cms.EDFilter("MixingModule",
    bunchspace = cms.int32(25)
)

process.o1 = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('my_output.root')
)

process.p1 = cms.Path(process.VtxSmeared*process.g4SimHits)
process.outpath = cms.EndPath(process.o1)
process.VtxSmeared.SigmaX = 0.00001
process.VtxSmeared.SigmaY = 0.00001
process.VtxSmeared.SigmaZ = 0.00001
process.g4SimHits.UseMagneticField = False


