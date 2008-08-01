import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")
process.load("SimGeneral.HepPDTESSource.pdt_cfi")

process.load("SimG4CMS.HcalTestBeam.TB2007GeometryXML_cfi")

process.load("Configuration.EventContent.EventContent_cff")

process.load("SimG4Core.Application.g4SimHits_cfi")

process.TFileService = cms.Service("TFileService",
    fileName = cms.string('hcaltb07.root')
)

process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('cout'),
    categories = cms.untracked.vstring('CaloSim', 
        'EcalGeom', 
        'EcalSim', 
        'HCalGeom', 
        'HcalSim', 
        'HcalTBSim', 
        'SimHCalData', 
        'SimG4CoreGeometry', 
        'SimG4CoreApplication', 
        'VertexGenerator'),
    debugModules = cms.untracked.vstring('*'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('DEBUG'),
        INFO = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        DEBUG = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        VertexGenerator = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        SimG4CoreApplication = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        SimG4CoreGeometry = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        EcalGeom = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        HCalGeom = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        CaloSim = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        EcalSim = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        HcalSim = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        HcalTBSim = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        SimHCalData = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        )
    )
)

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    moduleSeeds = cms.PSet(
        g4SimHits = cms.untracked.uint32(9876),
        VtxSmeared = cms.untracked.uint32(123456789)
    ),
    sourceSeed = cms.untracked.uint32(135799753)
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
        MinE = cms.untracked.double(9.99),
        MaxE = cms.untracked.double(10.01),
        PartID = cms.untracked.vint32(11)
    ),
    Verbosity = cms.untracked.int32(0)
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
    input = cms.untracked.int32(5)
)

process.o1 = cms.OutputModule("PoolOutputModule",
    process.FEVTSIMEventContent,
    fileName = cms.untracked.string('sim2007.root')
)

process.common_heavy_suppression1 = cms.PSet(
    NeutronThreshold = cms.double(30.0),
    ProtonThreshold = cms.double(30.0),
    IonThreshold = cms.double(30.0)
)
process.Timing = cms.Service("Timing")

process.p1 = cms.Path(process.VtxSmeared*process.g4SimHits)
process.outpath = cms.EndPath(process.o1)
process.g4SimHits.UseMagneticField = False
process.g4SimHits.Physics.type = 'SimG4Core/Physics/QGSP'
process.g4SimHits.StackingAction = cms.PSet(
    process.common_heavy_suppression1,
    TrackNeutrino = cms.bool(False),
    KillHeavy = cms.bool(False),
    SavePrimaryDecayProductsAndConversions = cms.untracked.bool(False)
)
process.g4SimHits.CaloSD = cms.PSet(
    process.common_beam_direction_parameters,
    process.common_heavy_suppression1,
    EminTrack      = cms.double(1.0),
    SuppressHeavy  = cms.bool(False),
    TmaxHit        = cms.double(1000.0),
    DetailedTiming = cms.untracked.bool(False),
    Verbosity      = cms.untracked.int32(0),
    CheckHits      = cms.untracked.int32(25),
    CorrectTOFBeam = cms.untracked.bool(False),
    UseMap         = cms.untracked.bool(True)
)
process.g4SimHits.ECalSD.UseBirkLaw = False
process.g4SimHits.ECalSD.BirkC1 = 0.013
process.g4SimHits.ECalSD.BirkC2 = '9.6e-6'
process.g4SimHits.ECalSD.SlopeLightYield = 0.02
process.g4SimHits.ECalSD.TestBeam = True
process.g4SimHits.HCalSD.UseBirkLaw = False
process.g4SimHits.HCalSD.BirkC1 = 0.013
process.g4SimHits.HCalSD.BirkC2 = '9.6e-6'
process.g4SimHits.HCalSD.UseShowerLibrary = False
process.g4SimHits.HCalSD.TestNumberingScheme = True
process.g4SimHits.HCalSD.UseHF = False
process.g4SimHits.HCalSD.ForTBH2 = True
process.g4SimHits.CaloTrkProcessing.TestBeam = True
process.g4SimHits.Watchers = cms.VPSet(cms.PSet(
    HcalTB06Analysis = cms.PSet(
        process.common_beam_direction_parameters,
        Names    = cms.vstring('HcalHits', 'EcalHitsEB'),
        EHCalMax = cms.untracked.double(2.0),
        ETtotMax = cms.untracked.double(20.0),
        Verbose  = cms.untracked.bool(True)
    ),
    type = cms.string('HcalTB06Analysis')
))

