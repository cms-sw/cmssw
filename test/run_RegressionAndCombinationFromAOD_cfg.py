import FWCore.ParameterSet.Config as cms

process = cms.Process("ExREG")
process.load("Configuration.StandardSequences.Services_cff")
process.load('Configuration.Geometry.GeometryDB_cff')
process.load('Configuration/StandardSequences/MagneticField_38T_cff')
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.GlobalTag.globaltag = 'START53_V10::All'

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    calibratedElectrons = cms.PSet(
        initialSeed = cms.untracked.uint32(1),
        engineName = cms.untracked.string('TRandom3')
    ),
)

process.load("EgammaAnalysis.ElectronTools.calibratedElectrons_cfi")

# dataset to correct
process.calibratedElectrons.isMC = cms.bool(True)
process.calibratedElectrons.inputDataset = cms.string("Summer12_DR53X_HCP2012")
process.calibratedElectrons.updateEnergyError = cms.bool(True)
process.calibratedElectrons.applyCorrections = cms.int32(1)
process.calibratedElectrons.verbose = cms.bool(True)
process.calibratedElectrons.synchronization = cms.bool(True)



process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
    )


process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring(
    #'/store/relval/CMSSW_5_3_4_cand1/RelValZEE/GEN-SIM-RECO/PU_START53_V10-v1/0003/0CBBC6C2-42F7-E111-B1C8-0030486780B4.root'
    '/store/relval/CMSSW_5_3_6/RelValZEE/GEN-SIM-RECO/PU_START53_V14-v1/0003/2C92DB85-E82C-E211-B8DE-003048D37560.root' 
        ))




process.load('EgammaAnalysis.ElectronTools.electronRegressionEnergyProducer_cfi')
process.eleRegressionEnergy.inputElectronsTag = cms.InputTag('gsfElectrons')
process.eleRegressionEnergy.inputCollectionType = cms.uint32(0)
process.eleRegressionEnergy.useRecHitCollections = cms.bool(True)
process.eleRegressionEnergy.produceValueMaps = cms.bool(True)

process.out = cms.OutputModule("PoolOutputModule",
                               outputCommands = cms.untracked.vstring('drop *',
                                                                      'keep *_*_*_ExREG'),
                               fileName = cms.untracked.string('electrons.root')
                                                              )
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.p = cms.Path(process.eleRegressionEnergy * process.calibratedElectrons)
#process.p = cms.Path(process.eleRegressionEnergy )
process.outpath = cms.EndPath(process.out)


