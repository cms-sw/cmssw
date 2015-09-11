import os
import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

### RANDOM setting (change last digit(s) to make runs different !)
process.load("Configuration.StandardSequences.SimulationRandomNumberGeneratorSeeds_cff")
#process.RandomNumberGeneratorService.generator.initialSeed = 12345XXXX


process.load("Configuration.StandardSequences.Simulation_cff")
process.load("SimGeneral.MixingModule.mixNoPU_cfi")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load('Configuration/StandardSequences/DigiToRaw_cff')
process.load('Configuration/StandardSequences/RawToDigi_cff')
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['mc']

process.load("IOMC.EventVertexGenerators.VtxSmearedGauss_cfi")
process.load("Configuration.StandardSequences.GeometryECALHCAL_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.g4SimHits.UseMagneticField = False

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 100

process.load("DQMServices.Core.DQM_cfg")
process.DQM.collectorHost = ''

process.options = cms.untracked.PSet(
    Rethrow = cms.untracked.vstring('ProductNotFound')
)

# Input source
process.source = cms.Source("PoolSource",
    firstEvent = cms.untracked.uint32(1),
    noEventSort = cms.untracked.bool(True),	
    duplicateCheckMode = cms.untracked.string("noDuplicateCheck"),
    fileNames = cms.untracked.vstring(
'file:/afs/cern.ch/cms/data/CMSSW/Validation/HcalHits/data/3_1_X/mc_nue.root'
    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)

process.FEVT = cms.OutputModule("PoolOutputModule",
     outputCommands = cms.untracked.vstring('drop *', 'keep *_MEtoEDMConverter_*_*'),
     fileName = cms.untracked.string("HcalValHarvestingEDM.root")
)

#-----------------------------------------------------------------------------
#                     Ananlyser + slient -> DQM
#-----------------------------------------------------------------------------
process.load('Configuration/StandardSequences/EDMtoMEAtRunEnd_cff')
process.dqmSaver.referenceHandling = cms.untracked.string('all')

cmssw_version = os.environ.get('CMSSW_VERSION','CMSSW_X_Y_Z')
Workflow = '/HcalValidation/'+'Harvesting/'+str(cmssw_version)
process.dqmSaver.workflow = Workflow

process.hcalRecoAnalyzer = cms.EDAnalyzer("HcalRecHitsValidation",
    outputFile = cms.untracked.string('HcalRecHitsValidation_ZS.root'),
    HBHERecHitCollectionLabel = cms.untracked.InputTag("hbhereco"),
    HFRecHitCollectionLabel   = cms.untracked.InputTag("hfreco"),
    HORecHitCollectionLabel   = cms.untracked.InputTag("horeco"),
    eventype = cms.untracked.string('single'),
    mc = cms.untracked.string('yes'),
    sign = cms.untracked.string('*'),
    hcalselector = cms.untracked.string('noise'),
    ecalselector = cms.untracked.string('no'),
    useAllHistos              = cms.untracked.bool(True) 
)
process.hcalrechitsClient = cms.EDAnalyzer("HcalRecHitsClient", 
     outputFile = cms.untracked.string('HcalRecHitsHarvestingME.root'),
     DQMDirName = cms.string("/") # root directory
)


#------------------------------------

process.simHcalDigis.HBlevel = -1000
process.simHcalDigis.HElevel = -1000
process.simHcalDigis.HOlevel = -1000
process.simHcalDigis.HFlevel = -1000
process.simHcalDigis.useConfigZSvalues = 1

process.VtxSmeared.SigmaX = 0.00001
process.VtxSmeared.SigmaY = 0.00001
process.VtxSmeared.SigmaZ = 0.00001


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

#--- Just a reminder : calolocalreco = cms.Sequence(ecalLocalRecoSequence+hcalLocalRecoSequence)
# RecoLocalCalo.Configuration.ecalLocalRecoSequence_cff
# RecoLocalCalo.Configuration.hcalLocalReco_cff
#--- but hcalLocalRecoSequence produces only hpheprereco!

#--- To cope with JP Chou pre-reco introduction to bring back hbhe RecHits collection to CaloTowers
#--- but better to include hcalGlobalRecoSequence in front of CaloTowers to bring hbherereco->hbhereco!
delattr(process,"hbhereco")
process.hbhereco = process.hbheprereco.clone()
process.hcalLocalRecoSequence.replace(process.hbheprereco,process.hbhereco)

#--- Special to exclude CASTOR 

process.mix.mixObjects.mixCH.mixCaloHits = cms.PSet(
    input = cms.VInputTag(cms.InputTag("g4SimHits","CaloHitsTk"), 
        cms.InputTag("g4SimHits","EcalHitsEB"), cms.InputTag("g4SimHits","EcalHitsEE"), cms.InputTag("g4SimHits","EcalHitsES"), cms.InputTag("g4SimHits","EcalTBH4BeamHits"), cms.InputTag("g4SimHits","HcalHits"), 
        cms.InputTag("g4SimHits","ZDCHITS")),
    type = cms.string('PCaloHit'),
    subdets = cms.vstring('CaloHitsTk', 
        'EcalHitsEB', 
        'EcalHitsEE', 
        'EcalHitsES', 
        'HcalHits', 
        'ZDCHITS')
)


process.g4SimHits.Generator.HepMCProductLabel = 'VtxSmeared'
process.p = cms.Path(
 process.VtxSmeared * process.g4SimHits * process.mix *
 process.ecalDigiSequence * process.hcalDigiSequence *
 process.ecalPacker *
 process.esDigiToRaw *
 process.hcalRawData *
 process.rawDataCollector *
 process.ecalDigis * 
 process.ecalPreshowerDigis * 
 process.hcalDigis *
 process.calolocalreco *
 process.hcalRecoAnalyzer *
 process.hcalrechitsClient * 
 process.dqmSaver)
