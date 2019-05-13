import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run2_2018_cff import Run2_2018
process = cms.Process('TEST',Run2_2018)

### RANDOM setting (change last digit(s) to make runs different !)
process.load("Configuration.StandardSequences.SimulationRandomNumberGeneratorSeeds_cff")

process.load("Configuration.StandardSequences.Simulation_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("SimGeneral.MixingModule.mixNoPU_cfi")
process.load('Configuration/StandardSequences/DigiToRaw_cff')
process.load('Configuration/StandardSequences/RawToDigi_cff')

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['phase1_2018_realistic']

process.load("IOMC.EventVertexGenerators.VtxSmearedGauss_cfi")
process.load("Configuration.StandardSequences.GeometryDB_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.g4SimHits.UseMagneticField = False

process.load("DQMServices.Core.DQMStore_cfi")
process.load("DQMServices.Components.MEtoEDMConverter_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000) 
)
# Input source
process.source = cms.Source("PoolSource",
    firstEvent = cms.untracked.uint32(XXXXX), 
    fileNames = cms.untracked.vstring('file:mc.root') 
) 

process.FEVT = cms.OutputModule("PoolOutputModule",
     outputCommands = cms.untracked.vstring('drop *', 'keep *_MEtoEDMConverter_*_*'),
     splitLevel = cms.untracked.int32(0),
     fileName = cms.untracked.string("output.root")
)

process.VtxSmeared.SigmaX = 0.00001
process.VtxSmeared.SigmaY = 0.00001
process.VtxSmeared.SigmaZ = 0.00001

process.load("Validation.HcalHits.HcalSimHitsValidation_cfi")
process.HcalSimHitsAnalyser.outputFile = cms.untracked.string('HcalSimHitsValidation.root')

process.load("Validation.HcalDigis.HcalDigisParam_cfi")
process.hcaldigisAnalyzer.outputFile = cms.untracked.string('HcalDigisValidationRelVal.root')

process.load("Validation.HcalRecHits.HcalRecHitParam_cfi")

process.load("Validation.CaloTowers.CaloTowersParam_cfi")
process.calotowersAnalyzer.outputFile = cms.untracked.string('CaloTowersValidationRelVal.root')



#------------- CUSTOMIZATION - replace hbhereco with hbheprereco
delattr(process,"hbhereco")
process.hbhereco = process.hbheprereco.clone()
process.hcalLocalRecoSequence = cms.Sequence(process.hbhereco+process.hfprereco+process.hfreco+process.horeco)


#---------- PATH
# -- NB: for vertex smearing the Label should be: "unsmeared" 
# for GEN produced since 760pre6, for older GEN - just "": 

process.VtxSmeared.src = cms.InputTag("generator", "") 
process.generatorSmeared = cms.EDProducer("GeneratorSmearedProducer")
process.g4SimHits.Generator.HepMCProductLabel = cms.InputTag('VtxSmeared')


process.p = cms.Path(
 process.VtxSmeared *
 process.generatorSmeared *
 process.g4SimHits *
 process.mix *
 process.ecalDigiSequence * 
 process.hcalDigiSequence *
 process.addPileupInfo *
 process.bunchSpacingProducer *
 process.ecalPacker *
 process.esDigiToRaw *
 process.hcalRawData *
 process.rawDataCollector *
 process.ecalDigis *
 process.ecalPreshowerDigis *
 process.hcalDigis *
 process.castorDigis *
 process.calolocalreco *
 process.caloTowersRec *
 process.hcalnoise *
 process.HcalSimHitsAnalyser *
 process.hcaldigisAnalyzer *
 process.calotowersAnalyzer *
 process.hcalRecoAnalyzer *
 process.MEtoEDMConverter
)

process.outpath = cms.EndPath(process.FEVT)

