# Runs PFBenchmarkAnalyzer and PFJetBenchmark on PFJet sample to
# monitor performance of PFJets

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("DQMServices.Core.DQM_cfg")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'IDEAL_30X::All'
process.load("RecoTauTag.Configuration.RecoPFTauTag_cff")
process.load("Configuration.StandardSequences.GeometryPilot2_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
#

#process.load("Validation.RecoParticleFlow.ztt_cfi")
#process.load("Validation.RecoParticleFlow.singletau_cfi")

#process.source = cms.Source("PoolSource",
#                            fileNames = cms.untracked.vstring('file:aod.root'
#                                                              )
  
process.source = cms.Source("PoolSource",
#                            fileNames = cms.untracked.vstring('rfio:/castor/cern.ch/user/g/gennai/CMSSW_310_pre2/ZTT_fastsim.root'
#                            fileNames = cms.untracked.vstring('file:/tmp/ZTT_fastsim.root'
                            fileNames = cms.untracked.vstring('file:aod.root'
                                                              )
                            )




process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)


process.load("Validation.RecoParticleFlow.tauBenchmarkGeneric_cff")
process.pfRecoTauProducerHighEfficiency.TrackerSignalConeSize_max = cms.double(0.5)

process.p =cms.Path(
    #process.pfRecoTauProducerHighEfficiency + 
    process.tauBenchmarkGeneric
    )


process.out = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('keep *'),
    fileName = cms.untracked.string('tree.root')
)
#process.outpath = cms.EndPath(process.out)

process.load("FWCore.MessageLogger.MessageLogger_cfi")


process.MessageLogger.cerr.FwkReport.reportEvery = 100

