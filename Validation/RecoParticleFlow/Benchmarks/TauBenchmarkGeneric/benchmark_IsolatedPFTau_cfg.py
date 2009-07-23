
# Runs PFBenchmarkAnalyzer and PFJetBenchmark on PFJet sample to
# monitor performance of PFJets

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("DQMServices.Core.DQM_cfg")

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
#'file:/storage/users/gennai/recoFastSimZTT_310pre11.root'

'rfio:/castor/cern.ch/user/g/gennai/CMSSW_310pre11/aod_ZTT_Full_1.root',
'rfio:/castor/cern.ch/user/g/gennai/CMSSW_310pre11/aod_ZTT_Full_2.root',
'rfio:/castor/cern.ch/user/g/gennai/CMSSW_310pre11/aod_ZTT_Full_3.root',
'rfio:/castor/cern.ch/user/g/gennai/CMSSW_310pre11/aod_ZTT_Full_4.root',
'rfio:/castor/cern.ch/user/g/gennai/CMSSW_310pre11/aod_ZTT_Full_5.root',
'rfio:/castor/cern.ch/user/g/gennai/CMSSW_310pre11/aod_ZTT_Full_6.root',
'rfio:/castor/cern.ch/user/g/gennai/CMSSW_310pre11/aod_ZTT_Full_7.root',
'rfio:/castor/cern.ch/user/g/gennai/CMSSW_310pre11/aod_ZTT_Full_8.root',
'rfio:/castor/cern.ch/user/g/gennai/CMSSW_310pre11/aod_ZTT_Full_9.root',
'rfio:/castor/cern.ch/user/g/gennai/CMSSW_310pre11/aod_ZTT_Full_10.root',
'rfio:/castor/cern.ch/user/g/gennai/CMSSW_310pre11/aod_ZTT_Full_11.root',
'rfio:/castor/cern.ch/user/g/gennai/CMSSW_310pre11/aod_ZTT_Full_12.root',
'rfio:/castor/cern.ch/user/g/gennai/CMSSW_310pre11/aod_ZTT_Full_13.root',
'rfio:/castor/cern.ch/user/g/gennai/CMSSW_310pre11/aod_ZTT_Full_14.root',
'rfio:/castor/cern.ch/user/g/gennai/CMSSW_310pre11/aod_ZTT_Full_15.root',
'rfio:/castor/cern.ch/user/g/gennai/CMSSW_310pre11/aod_ZTT_Full_16.root',
#'rfio:/castor/cern.ch/user/g/gennai/CMSSW_310pre11/aod_ZTT_Full_17.root',
'rfio:/castor/cern.ch/user/g/gennai/CMSSW_310pre11/aod_ZTT_Full_18.root',
'rfio:/castor/cern.ch/user/g/gennai/CMSSW_310pre11/aod_ZTT_Full_19.root',
'rfio:/castor/cern.ch/user/g/gennai/CMSSW_310pre11/aod_ZTT_Full_20.root',
'rfio:/castor/cern.ch/user/g/gennai/CMSSW_310pre11/aod_ZTT_Full_21.root',
'rfio:/castor/cern.ch/user/g/gennai/CMSSW_310pre11/aod_ZTT_Full_22.root',
#'rfio:/castor/cern.ch/user/g/gennai/CMSSW_310pre11/aod_ZTT_Full_23.root',
'rfio:/castor/cern.ch/user/g/gennai/CMSSW_310pre11/aod_ZTT_Full_24.root',
'rfio:/castor/cern.ch/user/g/gennai/CMSSW_310pre11/aod_ZTT_Full_25.root',
'rfio:/castor/cern.ch/user/g/gennai/CMSSW_310pre11/aod_ZTT_Full_26.root',
'rfio:/castor/cern.ch/user/g/gennai/CMSSW_310pre11/aod_ZTT_Full_27.root',
'rfio:/castor/cern.ch/user/g/gennai/CMSSW_310pre11/aod_ZTT_Full_28.root',
'rfio:/castor/cern.ch/user/g/gennai/CMSSW_310pre11/aod_ZTT_Full_29.root',
'rfio:/castor/cern.ch/user/g/gennai/CMSSW_310pre11/aod_ZTT_Full_30.root'


),
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string("noDuplicateCheck")

 )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
#For FastSim
process.load("Configuration.StandardSequences.Reconstruction_cff")
"""
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
#process.load("FastSimulation.Configuration.RandomServiceInitialization_cff")
#process.load("FastSimulation.Configuration.CommonInputs_cff")
#process.load("FastSimulation.Configuration.FamosSequences_cff")
"""
process.load("Configuration.StandardSequences.Services_cff")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = "MC_31X_V1::All"

#process.shrinkingConePFTauProducer.TrackerSignalConeSize_max = cms.double(0.2)
#process.shrinkingConePFTauDiscriminationByIsolationUsingLeadingPion.maxChargedPt = 0.5
#process.shrinkingConePFTauDiscriminationByIsolationUsingLeadingPion.maxGammaPt = 0.5

process.isolatedTaus = cms.EDFilter("PFTauSelector",
     src = cms.InputTag("shrinkingConePFTauProducer"),
     discriminators =  cms.VPSet(
                               cms.PSet( discriminator = cms.InputTag("shrinkingConePFTauDiscriminationByIsolationUsingLeadingPion"),selectionCut = cms.double(0.5))
)
)


#process.pfRecoTauTagInfoProducer.NeutrHadrCand_HcalclusminE = 0.4

process.load("Validation.RecoParticleFlow.tauBenchmarkIsolated_cff")
process.p =cms.Path(
    process.PFTau +
    process.isolatedTaus + 
    process.tauBenchmarkGeneric
    )


process.out = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('keep *'),
    fileName = cms.untracked.string('tree.root')
)
#process.outpath = cms.EndPath(process.out)

process.load("FWCore.MessageLogger.MessageLogger_cfi")


process.MessageLogger.cerr.FwkReport.reportEvery = 100

