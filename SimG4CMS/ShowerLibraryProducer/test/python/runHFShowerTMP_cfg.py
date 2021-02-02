import FWCore.ParameterSet.Config as cms

process = cms.Process("HFShowerLib")
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
#process.load("Geometry.HcalCommonData.hcalforwardshowerLong_cfi")
process.load("SimG4CMS.ShowerLibraryProducer.hcalforwardshower_cfi")
process.load("Geometry.HcalCommonData.hcalDDDSimConstants_cff")
process.load("IOMC.EventVertexGenerators.VtxSmearedGauss_cfi")
process.load('Configuration.StandardSequences.Generator_cff')
process.load('Configuration.StandardSequences.SimIdeal_cff')
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.Services_cff")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

if 'MessageLogger' in process.__dict__:
    process.MessageLogger.FiberSim=dict()
    process.MessageLogger.FlatThetaGun=dict()
    process.MessageLogger.HFShower=dict()
    process.MessageLogger.HcalForwardLib=dict()
    process.MessageLogger.SensitiveDetector=dict()

process.RandomNumberGeneratorService.generator.initialSeed = 12345

from IOMC.RandomEngine.RandomServiceHelper import RandomNumberServiceHelper
randSvc = RandomNumberServiceHelper(process.RandomNumberGeneratorService)
randSvc.populate()

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
    fileName = cms.untracked.string('simevent_50GeVElec.root')
)

process.TFileService = cms.Service("TFileService",
    fileName = cms.string('hfShowerLibSimu_extended2_50GeVElec_deneme.root')
)

# for GEN produced since 760pre6, for older GEN - just "":
process.VtxSmeared.src = cms.InputTag("generator", "unsmeared")
process.generatorSmeared = cms.EDProducer("GeneratorSmearedProducer")
process.g4SimHits.Generator.HepMCProductLabel = cms.InputTag('VtxSmeared')

process.p1 = cms.Path(
 process.generator *
 process.VtxSmeared *
 process.generatorSmeared *
 process.g4SimHits
)
process.outpath = cms.EndPath(process.o1)

process.g4SimHits.HCalSD.UseShowerLibrary = True
process.g4SimHits.HCalSD.UseParametrize = False
process.g4SimHits.HCalSD.UsePMTHits = False
process.g4SimHits.HCalSD.UseFibreBundleHits = False

process.g4SimHits.HFShower.UseShowerLibrary= True
process.g4SimHits.HFShower.UseR7600UPMT    = True
process.g4SimHits.HFShower.UseHFGflash = False
process.g4SimHits.HFShower.ApplyFiducialCut = False
process.g4SimHits.UseMagneticField = False

process.g4SimHits.NonBeamEvent = True
process.g4SimHits.Generator.ApplyPCuts   = False
process.g4SimHits.Generator.ApplyEtaCuts = False
process.g4SimHits.Physics.type = 'SimG4Core/Physics/FTFP_BERT_EMM'
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


