import FWCore.ParameterSet.Config as cms

process = cms.Process("CaloTest")
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

process.load("IOMC.EventVertexGenerators.VtxSmearedGauss_cfi")

process.load("Configuration.StandardSequences.MagneticField_cff")

process.load("Geometry.CMSCommonData.ecalhcalGeometryXML_cfi")

process.load("SimG4Core.Application.g4SimHits_cfi")

process.load("DQMServices.Core.DQM_cfg")

process.load("Validation.HcalHits.HcalHitValidation_cfi")

process.MessageLogger = cms.Service("MessageLogger",
    cout = cms.untracked.PSet(
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        HFShower = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        )
    ),
    categories = cms.untracked.vstring('HFShower'),
    destinations = cms.untracked.vstring('cout')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source("PoolSource",
    debugFlag = cms.untracked.bool(True),
    debugVebosity = cms.untracked.uint32(11),
    noEventSort = cms.untracked.bool(True),
    duplicateCheckMode = cms.untracked.string("noDuplicateCheck"),
     fileNames = cms.untracked.vstring('file:/afs/cern.ch/cms/data/CMSSW/Validation/HcalHits/data/3_1_X/mc_pi50_etaphi-344.root', 'file:/afs/cern.ch/cms/data/CMSSW/Validation/HcalHits/data/3_1_X/mc_pi50_etaphi+344.root')
)

process.Timing = cms.Service("Timing")

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    moduleSeeds = cms.PSet(
        g4SimHits = cms.untracked.uint32(9876),
        VtxSmeared = cms.untracked.uint32(123456789)
    ),
    sourceSeed = cms.untracked.uint32(135799753)
)

process.USER = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('simevent_HF.root')
)

process.p1 = cms.Path(process.VtxSmeared*process.g4SimHits*process.hcalHitValid)
process.outpath = cms.EndPath(process.USER)
process.VtxSmeared.SigmaX = 0.00001
process.VtxSmeared.SigmaY = 0.00001
process.VtxSmeared.SigmaZ = 0.00001
process.g4SimHits.UseMagneticField = False
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
process.DQM.collectorHost = ''
process.hcalHitValid.outputFile = 'valid_HF.root'


