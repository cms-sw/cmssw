import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run2_2018_cff import Run2_2018
process = cms.Process("CaloTest",Run2_2018)

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

process.load("IOMC.EventVertexGenerators.VtxSmearedGauss_cfi")
process.load('Configuration.StandardSequences.Generator_cff')

#Geometry
process.load("SimG4CMS.Calo.testGeometryPMTXML_cfi")
process.load("Geometry.EcalCommonData.ecalSimulationParameters_cff")
process.load("Geometry.HcalCommonData.hcalDDConstants_cff")

#Magnetic Field
process.load("Configuration.StandardSequences.MagneticField_cff")

process.load("Configuration.EventContent.EventContent_cff")

process.load("SimG4Core.Application.g4SimHits_cfi")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['run2_mc']

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        HcalSim = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        ValidHcal = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        enable = cms.untracked.bool(True)
    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)

process.source = cms.Source("EmptySource")

process.generator = cms.EDProducer("FlatRandomEGunProducer",
    PGunParameters = cms.PSet(
        PartID = cms.vint32(13),
        MinEta = cms.double(2.95),
        MaxEta = cms.double(3.30),
        MinPhi = cms.double(-3.1415926),
        MaxPhi = cms.double(3.1415926),
        MinE   = cms.double(99.99),
        MaxE   = cms.double(100.01)
    ),
    AddAntiParticle = cms.bool(False),
    Verbosity       = cms.untracked.int32(0),
    firstRun        = cms.untracked.uint32(1)
)

process.Timing = cms.Service("Timing")

process.load("IOMC.RandomEngine.IOMC_cff")
process.RandomNumberGeneratorService.generator.initialSeed = 456789
process.RandomNumberGeneratorService.g4SimHits.initialSeed = 9876
process.RandomNumberGeneratorService.VtxSmeared.initialSeed = 123456789
process.rndmStore = cms.EDProducer("RandomEngineStateProducer")

process.USER = cms.OutputModule("PoolOutputModule",
    process.FEVTSIMEventContent,
    fileName = cms.untracked.string('simevent_HFPMT.root')
)

process.p1 = cms.Path(process.generator*process.VtxSmeared*process.generatorSmeared*process.g4SimHits)
process.outpath = cms.EndPath(process.USER)
process.g4SimHits.Physics.type = 'SimG4Core/Physics/QGSP_BERT_EML'
process.g4SimHits.Physics.DefaultCutValue = 0.1
process.g4SimHits.HCalSD.UseShowerLibrary = False
process.g4SimHits.HCalSD.UseParametrize = True
process.g4SimHits.HCalSD.UsePMTHits = True
process.g4SimHits.OnlySDs = ['EcalSensitiveDetector', 'CaloTrkProcessing', 'HcalSensitiveDetector']
process.g4SimHits.Watchers = cms.VPSet(cms.PSet(
    SimG4HcalValidation = cms.PSet(
        TimeLowLimit = cms.double(0.0),
        LabelNxNInfo = cms.untracked.string('HcalInfoNxN'),
        LabelLayerInfo = cms.untracked.string('HcalInfoLayer'),
        HcalHitThreshold = cms.double(1e-20),
        Phi0 = cms.double(0.3054),
        ConeSize = cms.double(0.5),
        InfoLevel = cms.int32(2),
        JetThreshold = cms.double(5.0),
        EcalHitThreshold = cms.double(1e-20),
        TimeUpLimit = cms.double(999.0),
        HcalClusterOnly = cms.bool(False),
        Eta0 = cms.double(0.3045),
        LabelJetsInfo = cms.untracked.string('HcalInfoJets'),
        Names = cms.vstring('HcalHits', 
            'EcalHitsEB', 
            'EcalHitsEE', 
            'EcalHitsES'),
        HcalSampling = cms.bool(True)
    ),
    type = cms.string('SimG4HcalValidation')
))


