import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

### RANDOM setting (change last digit(s) to make runs different !)
process.load("Configuration.StandardSequences.SimulationRandomNumberGeneratorSeeds_cff")
#process.RandomNumberGeneratorService.generator.initialSeed = 12345XXXX

process.load("Configuration.StandardSequences.Simulation_cff")
process.load("Configuration.StandardSequences.MixingNoPileUp_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load('Configuration/StandardSequences/DigiToRaw_cff')
process.load('Configuration/StandardSequences/RawToDigi_cff')
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'MC_3XY_V14::All'

process.load("Configuration.StandardSequences.VtxSmearedGauss_cff")
process.load("Configuration.StandardSequences.GeometryECALHCAL_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.g4SimHits.UseMagneticField = False

process.load("DQMServices.Core.DQM_cfg")
process.DQM.collectorHost = ''

# Input source
process.source = cms.Source("PoolSource",
    firstEvent = cms.untracked.uint32(XXXXX),
    debugFlag = cms.untracked.bool(True),
    debugVebosity = cms.untracked.uint32(2),
    fileNames = cms.untracked.vstring('file:mc.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(2000)
)

process.VtxSmeared.SigmaX = 0.00001
process.VtxSmeared.SigmaY = 0.00001
process.VtxSmeared.SigmaZ = 0.00001

#process.hbhereco.digiLabel = 'simHcalDigis'
#process.horeco.digiLabel = 'simHcalDigis'
#process.hfreco.digiLabel = 'simHcalDigis'

#process.simHcalDigis.HBlevel = -1000
#process.simHcalDigis.HOlevel = -1000
#process.simHcalDigis.HElevel = 8
#process.simHcalDigis.HFlevel = 9

process.HcalSimHitsAnalyser = cms.EDFilter("HcalSimHitsValidation",
    outputFile = cms.untracked.string('HcalSimHitsValidation.root'),
)   

process.hcalDigiAnalyzer = cms.EDFilter("HcalDigiTester",
    digiLabel = cms.InputTag("simHcalUnsuppressedDigis"),
    outputFile = cms.untracked.string('HcalDigisValidationHB.root'),
    hcalselector = cms.untracked.string('HB'),
    zside = cms.untracked.string('*')
)

process.hcalRecoAnalyzer = cms.EDFilter("HcalRecHitsValidation",
    outputFile = cms.untracked.string('output.root'),
    eventype = cms.untracked.string('single'),
    mc = cms.untracked.string('yes'),
    sign = cms.untracked.string('*'),
    hcalselector = cms.untracked.string('all'),
    ecalselector = cms.untracked.string('yes')
)

process.hcalTowerAnalyzer = cms.EDAnalyzer("CaloTowersValidation",
    outputFile = cms.untracked.string('CaloTowersValidationHB.root'),
    CaloTowerCollectionLabel = cms.untracked.string('towerMaker'), # noHO!
    hcalselector = cms.untracked.string('all')
)


process.FEVT = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('full_output.root'),
    dataset = cms.untracked.PSet(
    dataTier = cms.untracked.string('GEN-SIM-RECO'),
    filterName = cms.untracked.string('')
    )
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
 process.HcalSimHitsAnalyser *
 process.hcalRecoAnalyzer *
 process.hcalTowerAnalyzer
)

### process.outpath = cms.EndPath(process.FEVT)

