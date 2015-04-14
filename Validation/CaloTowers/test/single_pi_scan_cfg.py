import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

### RANDOM setting (change last digit(s) to make runs different !)
process.load("Configuration.StandardSequences.SimulationRandomNumberGeneratorSeeds_cff")
#process.RandomNumberGeneratorService.generator.initialSeed = 12345XXXX

process.load("Configuration.StandardSequences.Simulation_cff")

#--- replacing two old includes with two new for 4_X_Y CMSSW
#process.load("Configuration.StandardSequences.MixingNoPileUp_cff")
#process.load("Configuration.StandardSequences.VtxSmearedGauss_cff")
process.load("SimGeneral.MixingModule.mixNoPU_cfi")
process.load("IOMC.EventVertexGenerators.VtxSmearedGauss_cfi")

process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load('Configuration/StandardSequences/DigiToRaw_cff')
process.load('Configuration/StandardSequences/RawToDigi_cff')
process.load("Configuration.StandardSequences.GeometryECALHCAL_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.g4SimHits.UseMagneticField = False
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['mc']

process.load("DQMServices.Core.DQM_cfg")
process.DQM.collectorHost = ''
process.load("DQMServices.Components.MEtoEDMConverter_cfi")

# Input source
process.source = cms.Source("PoolSource",
    firstEvent = cms.untracked.uint32(1),
    fileNames = cms.untracked.vstring('file:mc.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

process.FEVT = cms.OutputModule("PoolOutputModule",
     outputCommands = cms.untracked.vstring('drop *', 'keep *_MEtoEDMConverter_*_*'),
     fileName = cms.untracked.string("output.root")
)

process.VtxSmeared.SigmaX = 0.00001
process.VtxSmeared.SigmaY = 0.00001
process.VtxSmeared.SigmaZ = 0.00001

### ---- if unsuppressed Digi required (SimCalorimetry/HcalZeroSuppressionProducers/python/hcalDigisNoSuppression_cfi.py)
#process.simHcalDigis.markAndPass = cms.bool(False),
#process.simHcalDigis.    useConfigZSvalues = cms.int32(1)
#process.simHcalDigis.HBlevel = -999
#process.simHcalDigis.HOlevel = -999
#process.simHcalDigis.HElevel = -999
#process.simHcalDigis.HFlevel = -999


process.HcalSimHitsAnalyser = cms.EDAnalyzer("HcalSimHitsValidation",
    outputFile = cms.untracked.string('HcalSimHitsValidation.root')
)   

process.hcalRecoAnalyzer = cms.EDAnalyzer("HcalRecHitsValidation",
    outputFile                = cms.untracked.string('HcalRecHitValidationRelVal.root'),
    HBHERecHitCollectionLabel = cms.untracked.InputTag("hbhereco"),
    HFRecHitCollectionLabel   = cms.untracked.InputTag("hfreco"),
    HORecHitCollectionLabel   = cms.untracked.InputTag("horeco"),
    eventype                  = cms.untracked.string('single'),#multi if RelVal
    ecalselector              = cms.untracked.string('yes'),
    hcalselector              = cms.untracked.string('all'),
    mc                        = cms.untracked.string('yes')  # default
)

process.hcalTowerAnalyzer = cms.EDAnalyzer("CaloTowersValidation",
    outputFile               = cms.untracked.string('CaloTowersValidationRelVal.root'),
    CaloTowerCollectionLabel = cms.untracked.InputTag('towerMaker'),
    
    hcalselector             = cms.untracked.string('all'),
    mc                       = cms.untracked.string('yes')  # default!
)


### Special - CaloOnly ------------------------------------

#--- comes from DigiToRaw_cff.py
process.ecalPacker.Label = 'simEcalDigis'
process.ecalPacker.InstanceEB = 'ebDigis'
process.ecalPacker.InstanceEE = 'eeDigis'
process.ecalPacker.labelEBSRFlags = "simEcalDigis:ebSrFlags"
process.ecalPacker.labelEESRFlags = "simEcalDigis:eeSrFlags"
#
#- hcalRawData (EventFilter/HcalRawToDigi/python/HcalDigiToRaw_cfi.py
#                 uses simHcalDigis by default...

#--- to force RAW->Digi 
process.ecalDigis.InputLabel = 'rawDataCollector'
process.hcalDigis.InputLabel = 'rawDataCollector'
process.ecalPreshowerDigis.sourceTag = 'rawDataCollector'

#--- calolocalreco = cms.Sequence(ecalLocalRecoSequence+hcalLocalRecoSequence)
# RecoLocalCalo.Configuration.ecalLocalRecoSequence_cff
# RecoLocalCalo.Configuration.hcalLocalReco_cff

#--- To cope with pre-reco in 3_11_X introduction to bring back hbhe RecHits collection to CaloTowers
delattr(process,"hbhereco")
process.hbhereco = process.hbheprereco.clone()
process.hcalLocalRecoSequence.replace(process.hbheprereco,process.hbhereco)


process.g4SimHits.Generator.HepMCProductLabel = 'generator'
process.p = cms.Path(
 process.VtxSmeared * process.g4SimHits * process.mix *
 process.calDigi *
 process.ecalPacker *
 process.esDigiToRaw *
 process.hcalRawData *
 process.rawDataCollector *
 process.ecalDigis *
 process.ecalPreshowerDigis *
 process.hcalDigis *
 process.calolocalreco *
 process.caloTowersRec *
 process.hcalnoise *
 process.HcalSimHitsAnalyser *
 process.hcalTowerAnalyzer *
 process.hcalRecoAnalyzer *
 process.MEtoEDMConverter
)

process.outpath = cms.EndPath(process.FEVT)
