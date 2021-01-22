import FWCore.ParameterSet.Config as cms

process = cms.Process("Sim")
process.load("SimG4CMS.Calo.PythiaMinBias_cfi")
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.load("Configuration.Geometry.GeometryExtended2017Reco_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.EventContent.EventContent_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['phase1_2017_design']

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
        'root://xrootd.unl.edu//store/mc/Phys14DR/DYToMuMu_M-50_Tune4C_13TeV-pythia8/GEN-SIM-RECO/PU20bx25_tsg_castor_PHYS14_25_V1-v1/10000/184C1AC9-A775-E411-9196-002590200824.root'
#        'file:/afs/cern.ch/user/a/amkalsi/public/ForSunandaDa/024A536E-48EE-E611-843A-001E67E71C95.root'
        )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(200)
)

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        HitStudy = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        enable = cms.untracked.bool(True)
    )
)

process.load("IOMC.RandomEngine.IOMC_cff")
process.RandomNumberGeneratorService.generator.initialSeed = 456789
process.RandomNumberGeneratorService.g4SimHits.initialSeed = 9876
process.RandomNumberGeneratorService.VtxSmeared.initialSeed = 123456789
process.rndmStore = cms.EDProducer("RandomEngineStateProducer")

process.analyze = cms.EDAnalyzer("HcalSimHitDump",
                                 ModuleLabel  = cms.untracked.string('g4SimHits'),
                                 HCCollection = cms.untracked.string('HcalHits'),
                                 MaxEvent     = cms.untracked.int32(20)
                                 )
 
process.schedule = cms.Path(process.analyze)

