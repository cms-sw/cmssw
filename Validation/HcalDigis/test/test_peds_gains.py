import FWCore.ParameterSet.Config as cms

process = cms.Process("VALID")

process.load("Configuration.StandardSequences.SimulationRandomNumberGeneratorSeeds_cff")
process.load("Configuration.StandardSequences.Simulation_cff")
process.load("Configuration.StandardSequences.MixingNoPileUp_cff")
process.load("Configuration.StandardSequences.VtxSmearedGauss_cff")
process.load("Configuration.StandardSequences.GeometryHCAL_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['mc']

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.load("DQMServices.Core.DQM_cfg")
process.DQM.collectorHost = ''

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
'file:/afs/cern.ch/cms/data/CMSSW/Validation/HcalHits/data/3_1_X/mc_nue.root'
    )
)

process.hcalDigiAnalyzer = cms.EDAnalyzer("HcalDigiTester",
    digiLabel = cms.InputTag("simHcalUnsuppressedDigis"),
    outputFile = cms.untracked.string('HcalDigisValidation.root'),
    hcalselector = cms.untracked.string('noise'),
    zside = cms.untracked.string('*')
)

process.g4SimHits.Generator.HepMCProductLabel = 'generator'
process.p = cms.Path(
process.VtxSmeared * 
process.g4SimHits * 
process.mix * 
process.simHcalUnsuppressedDigis *
process.simHcalDigis *
process.hcalDigiAnalyzer)
