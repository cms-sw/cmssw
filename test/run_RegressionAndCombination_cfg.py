import FWCore.ParameterSet.Config as cms

process = cms.Process("ExREG")
process.load("Configuration.StandardSequences.Services_cff")
process.load('Configuration.Geometry.GeometryDB_cff')
process.load('Configuration/StandardSequences/MagneticField_38T_cff')
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.GlobalTag.globaltag = 'GR_P_V42_AN3::All'

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    calibratedPatElectrons = cms.PSet(
        initialSeed = cms.untracked.uint32(1),
        engineName = cms.untracked.string('TRandom3')
    ),
)

process.load("EgammaAnalysis.ElectronTools.calibratedPatElectrons_cfi")

# dataset to correct
process.calibratedPatElectrons.isMC = cms.bool(False)
process.calibratedPatElectrons.inputDataset = cms.string("22Jan2013ReReco")
process.calibratedPatElectrons.updateEnergyError = cms.bool(True)
process.calibratedPatElectrons.correctionsType = cms.int32(2)
process.calibratedPatElectrons.combinationType = cms.int32(3)
process.calibratedPatElectrons.lumiRatio = cms.double(1.0)
process.calibratedPatElectrons.verbose = cms.bool(True)
process.calibratedPatElectrons.synchronization = cms.bool(True)



process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
    )


process.source = cms.Source("PoolSource",
    #fileNames = cms.untracked.vstring('file:../../../PhysicsTools/PatAlgos/test/patTuple_standard.root')
    #fileNames = cms.untracked.vstring('file:patTuple_standard.root')
    #fileNames = cms.untracked.vstring('/store/cernproduction/hzz4l/CMG/DoubleElectron/Run2012D-PromptReco-v1/AOD/PAT_CMG_V5_10_0/cmgTuple_1.root')
    #fileNames = cms.untracked.vstring('root://eoscms//eos/cms/store/cmst3/user/cmgtools/CMG/DoubleElectron/Run2012A-13Jul2012-v1/AOD/V5/PAT_CMG_V5_10_0/cmgTuple_305.root')
    fileNames = cms.untracked.vstring('/store/cernproduction/hzz4l/CMG/DoubleElectron/Run2012B-13Jul2012-v1/AOD/V5/PAT_CMG_V5_10_0/cmgTuple_1.root')
    )


process.load('EgammaAnalysis.ElectronTools.electronRegressionEnergyProducer_cfi')
process.eleRegressionEnergy.inputElectronsTag = cms.InputTag('patElectronsWithTrigger')
#process.eleRegressionEnergy.inputCollectionType = cms.uint32(0)
#process.eleRegressionEnergy.useRecHitCollections = cms.bool(True)
process.eleRegressionEnergy.regressionInputFile = cms.string("EgammaAnalysis/ElectronTools/data/eleEnergyRegWeights_WithSubClusters_VApr15.root")
process.eleRegressionEnergy.energyRegressionType = cms.uint32(2)

process.out = cms.OutputModule("PoolOutputModule",
                               outputCommands = cms.untracked.vstring('drop *',
                                                                      'keep *_*_*_ExREG'),
                               fileName = cms.untracked.string('electrons_PAT.root')
                                                              )
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.p = cms.Path(process.eleRegressionEnergy * process.calibratedPatElectrons)
process.outpath = cms.EndPath(process.out)


