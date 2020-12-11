import FWCore.ParameterSet.Config as cms

process = cms.Process("HFShowerLib")
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

process.load("IOMC.EventVertexGenerators.VtxSmearedGauss_cfi")

#process.load("Geometry.HcalCommonData.hcalforwardshowerLong_cfi")
process.load("SimG4CMS.ShowerLibraryProducer.hcalforwardshower_cfi")
process.load("SimG4Core.Application.g4SimHits_cfi")

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        DEBUG = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        FiberSim = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        FlatThetaGun = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        G4cerr = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        G4cout = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        HFShower = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        HcalForwardLib = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        INFO = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        enable = cms.untracked.bool(True),
        threshold = cms.untracked.string('DEBUG')
    ),
    debugModules = cms.untracked.vstring('*')
)

process.load("Configuration.StandardSequences.Services_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    moduleSeeds = cms.PSet(
        g4SimHits = cms.untracked.uint32(12341),
        VtxSmeared = cms.untracked.uint32(39712),
        generator = cms.untracked.uint32(23451)
    ),
    sourceSeed = cms.untracked.uint32(24124),
)

process.Timing = cms.Service("Timing")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.source = cms.Source("EmptySource")

process.generator = cms.EDProducer("FlatRandomEThetaGunProducer",
    PGunParameters = cms.PSet(
        PartID   = cms.vint32(11),
        #MinTheta = cms.double(-1.145762838),
        #MaxTheta = cms.double(1.145762838),
        MinTheta = cms.double(0.019997),
        MaxTheta = cms.double(0.019997),
        MinPhi   = cms.double(3.14500926),
        MaxPhi   = cms.double(3.14500926),
        MinE     = cms.double(100.0),
        MaxE     = cms.double(100.0)
    ),
    Verbosity = cms.untracked.int32(2),
    AddAntiParticle = cms.bool(False),
    firstRun = cms.untracked.uint32(1)
)

process.o1 = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('/tmp/myucel/simevent_50GeVElec.root')
)

process.TFileService = cms.Service("TFileService",
    fileName = cms.string('/tmp/myucel/hfShowerLibSimu_extended2_50GeVElec_deneme.root')
)

process.p1 = cms.Path(cms.SequencePlaceholder("randomEngineStateProducer")+process.generator*process.VtxSmeared*process.g4SimHits)
process.outpath = cms.EndPath(process.o1)

process.g4SimHits.HCalSD.UseShowerLibrary = True
process.g4SimHits.HCalSD.UseParametrize = False
process.g4SimHits.HCalSD.UsePMTHits = False
process.g4SimHits.HCalSD.UseFibreBundleHits = False

process.g4SimHits.HFShower.UseShowerLibrary= True
process.g4SimHits.HFShower.UseR7600UPMT    = True
process.g4SimHits.HFShower.UseHFGflash = False
process.g4SimHits.HFShower.ApplyFiducialCut = False


process.g4SimHits.NonBeamEvent = True
process.g4SimHits.Generator.ApplyPCuts   = False
process.g4SimHits.Generator.ApplyEtaCuts = False
process.g4SimHits.Physics.type = 'SimG4Core/Physics/LHEP_EMV'
process.g4SimHits.Physics.DefaultCutValue = 0.1
process.g4SimHits.G4Commands = ['/tracking/verbose 1']
process.g4SimHits.Watchers = cms.VPSet(cms.PSet(
    HFShowerLibraryProducer = cms.PSet(
        Names = cms.vstring('FibreHits', 
            'ChamberHits', 
            'WedgeHits')
    ),
    type = cms.string('HcalForwardAnalysis')
))


