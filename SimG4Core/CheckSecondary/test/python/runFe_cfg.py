import FWCore.ParameterSet.Config as cms

process = cms.Process("CaloTest")
process.load("SimGeneral.HepPDTESSource.pdt_cfi")

process.load("IOMC.EventVertexGenerators.VtxSmearedGauss_cfi")

process.load("SimG4Core.CheckSecondary.FeTarget_cfi")

process.load("SimG4Core.Application.g4SimHits_cfi")

process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('cout'),
    categories = cms.untracked.vstring('SimG4CoreApplication', 'CheckSecondary'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO'),
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        CheckSecondary = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        SimG4CoreApplication = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        )
    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(20000)
)

process.source = cms.Source("EmptySource")

process.generator = cms.EDProducer("FlatRandomEGunProducer",
    PGunParameters = cms.PSet(
        MinEta = cms.double(0.0),
        MaxEta = cms.double(0.0),
        MinPhi = cms.double(1.57079632679),
        MaxPhi = cms.double(1.57079632679),
#        PartID = cms.vint32(-211),
#	MinE   = cms.double(1.010),
#	MaxE   = cms.double(1.010)
#	MinE   = cms.double(2.005),
#	MaxE   = cms.double(2.005)
#	MinE   = cms.double(3.003),
#	MaxE   = cms.double(3.003)
#	MinE   = cms.double(4.002),
#	MaxE   = cms.double(4.002)
#	MinE   = cms.double(5.002),
#	MaxE   = cms.double(5.002)
#	MinE   = cms.double(6.002),
#	MaxE   = cms.double(6.002)
#	MinE   = cms.double(7.001),
#	MaxE   = cms.double(7.001)
#	MinE   = cms.double(8.001),
#	MaxE   = cms.double(8.001)
#	MinE   = cms.double(9.001),
#	MaxE   = cms.double(9.001)
#	MinE   = cms.double(10.001),
#	MaxE   = cms.double(10.001)
#	MinE   = cms.double(15.001),
#	MaxE   = cms.double(15.001)
#	MinE   = cms.double(20.001),
#	MaxE   = cms.double(20.001)
#	MinE   = cms.double(30.001),
#	MaxE   = cms.double(30.001)
#	MinE   = cms.double(50.001),
#	MaxE   = cms.double(50.001)
#	MinE   = cms.double(100.0),
#	MaxE   = cms.double(100.0)
#	MinE   = cms.double(150.0),
#	MaxE   = cms.double(150.0)
#	MinE   = cms.double(200.0),
#	MaxE   = cms.double(200.0)
#	MinE   = cms.double(300.0),
#	MaxE   = cms.double(300.0)
        PartID = cms.vint32(2212),
#	MinE   = cms.double(1.3713),
#	MaxE   = cms.double(1.3713)
	MinE   = cms.double(2.2092),
	MaxE   = cms.double(2.2092)
#	MinE   = cms.double(3.1433),
#	MaxE   = cms.double(3.1433)
#	MinE   = cms.double(4.1086),
#	MaxE   = cms.double(4.1086)
#	MinE   = cms.double(5.0873),
#	MaxE   = cms.double(5.0873)
#	MinE   = cms.double(6.0729),
#	MaxE   = cms.double(6.0729)
#	MinE   = cms.double(7.0626),
#	MaxE   = cms.double(7.0626)
#	MinE   = cms.double(8.0548),
#	MaxE   = cms.double(8.0548)
#	MinE   = cms.double(9.0488),
#	MaxE   = cms.double(9.0488)
#	MinE   = cms.double(10.0439),
#	MaxE   = cms.double(10.0439)
#	MinE   = cms.double(15.0293),
#	MaxE   = cms.double(15.0293)
#	MinE   = cms.double(20.0220),
#	MaxE   = cms.double(20.0220)
#	MinE   = cms.double(30.0147),
#	MaxE   = cms.double(30.0147)
#	MinE   = cms.double(50.0088),
#	MaxE   = cms.double(50.0088)
#	MinE   = cms.double(100.0044),
#	MaxE   = cms.double(100.0044)
#	MinE   = cms.double(150.0029),
#	MaxE   = cms.double(150.0029)
# 	MinE   = cms.double(200.0022),
#	MaxE   = cms.double(200.0022)
#	MinE   = cms.double(300.001),
#	MaxE   = cms.double(300.001)
    ),
    Verbosity = cms.untracked.int32(0),
    AddAntiParticle = cms.bool(False),
    firstRun        = cms.untracked.uint32(1)
)

process.Timing = cms.Service("Timing")

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    moduleSeeds = cms.PSet(
        generator = cms.untracked.uint32(456789),
        g4SimHits = cms.untracked.uint32(9876),
        VtxSmeared = cms.untracked.uint32(123456789)
    ),
    sourceSeed = cms.untracked.uint32(135799753)
)

process.p1 = cms.Path(process.generator*process.VtxSmeared*process.g4SimHits)
process.VtxSmeared.SigmaX = 0.00001
process.VtxSmeared.SigmaY = 0.00001
process.VtxSmeared.SigmaZ = 0.00001
process.g4SimHits.UseMagneticField     = False
process.g4SimHits.Physics.type         = 'SimG4Core/Physics/QGSP'
process.g4SimHits.Physics.EMPhysics    = False
process.g4SimHits.Watchers = cms.VPSet(cms.PSet(
    CheckSecondary = cms.PSet(
        SaveInFile = cms.untracked.string('FeQGSP_pro_2.0GeV.root'),
        Verbosity = cms.untracked.int32(0),
        MinimumDeltaE = cms.untracked.double(0.0),
        KillAfter = cms.untracked.int32(1)
    ),
    type = cms.string('CheckSecondary')
), 
    cms.PSet(
        type = cms.string('KillSecondariesRunAction')
    ))

