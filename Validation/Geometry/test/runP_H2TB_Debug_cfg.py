import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

#Geometry
#
process.load("SimG4CMS.HcalTestBeam.TB2007GeometryXML_cfi")

# Detector simulation (Geant4-based)
#
process.load("SimG4Core.Application.g4SimHits_cfi")

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    moduleSeeds = cms.PSet(
        g4SimHits = cms.untracked.uint32(9876),
        VtxSmeared = cms.untracked.uint32(123456789)
    ),
    sourceSeed = cms.untracked.uint32(135799753)
)

process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('cout'),
    categories = cms.untracked.vstring('MaterialBudget'),
    debugModules = cms.untracked.vstring('*'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('DEBUG'),
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        MaterialBudget = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        )
    )
)
process.common_beam_direction_parameters = cms.PSet(
    MinEta = cms.untracked.double(1.562),
    MaxEta = cms.untracked.double(1.562),
    MinPhi = cms.untracked.double(0.0436),
    MaxPhi = cms.untracked.double(0.0436),
    BeamPosition = cms.untracked.double(-800.0)
)

process.source = cms.Source("FlatRandomEGunSource",
    PGunParameters = cms.untracked.PSet(
        process.common_beam_direction_parameters,
        PartID = cms.untracked.vint32(14),
        MinE = cms.untracked.double(9.99),
        MaxE = cms.untracked.double(10.01)
    ),
    Verbosity = cms.untracked.int32(0),
    AddAntiParticle = cms.untracked.bool(False),
    firstRun = cms.untracked.uint32(1)
)

from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import *
process.VtxSmeared = cms.EDFilter("BeamProfileVtxGenerator",
    process.common_beam_direction_parameters,
    VtxSmearedCommon,
    BeamMeanX = cms.untracked.double(0.0),
    BeamMeanY = cms.untracked.double(0.0),
    BeamSigmaX = cms.untracked.double(0.0001),
    BeamSigmaY = cms.untracked.double(0.0001),
    GaussianProfile = cms.untracked.bool(False),
    BinX = cms.untracked.int32(50),
    BinY = cms.untracked.int32(50),
    File = cms.untracked.string('beam.profile'),
    UseFile = cms.untracked.bool(False),
    TimeOffset = cms.double(0.)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.TFileService = cms.Service("TFileService",
    fileName = cms.string('matbdg_HCAL1.root')
)

process.p1 = cms.Path(process.VtxSmeared*process.g4SimHits)
process.g4SimHits.Generator.HepMCProductLabel = 'source'
process.g4SimHits.NonBeamEvent = True
process.g4SimHits.UseMagneticField = False
process.g4SimHits.Physics.type = 'SimG4Core/Physics/DummyPhysics'
process.g4SimHits.Physics.DummyEMPhysics = True
process.g4SimHits.Physics.CutsPerRegion = False
process.g4SimHits.ECalSD.TestBeam = True
process.g4SimHits.HCalSD.UseShowerLibrary = False
process.g4SimHits.HCalSD.TestNumberingScheme = True
process.g4SimHits.HCalSD.UseHF = False
process.g4SimHits.HCalSD.ForTBH2 = True
process.g4SimHits.CaloTrkProcessing.TestBeam = True
process.g4SimHits.Watchers = cms.VPSet(cms.PSet(
    MaterialBudgetHcal = cms.PSet(
        FillHisto = cms.untracked.bool(False),
        NBinPhi   = cms.untracked.int32(180),
        NBinEta   = cms.untracked.int32(100),
        MaxEta    = cms.untracked.double(3.0),
        EtaLow    = cms.untracked.double(-3.0),
        EtaHigh   = cms.untracked.double(3.0),
        RMax      = cms.untracked.double(10.0),
        ZMax      = cms.untracked.double(15.0)
    ),
    type = cms.string('MaterialBudgetHcal')
))


