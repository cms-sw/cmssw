# Runs PFBenchmarkAnalyzer and PFJetBenchmark on PFJet sample to
# monitor performance of PFJets

import FWCore.ParameterSet.Config as cms
  
process = cms.Process("TEST")
process.load("DQMServices.Core.DQM_cfg")


process.source = cms.Source (
    "PoolSource",    
    fileNames = cms.untracked.vstring(
    # Fast
       #'rfio:/castor/cern.ch/user/p/pjanot/CMSSW310pre11/aod_QCDForPF_Fast_0.root',
       #'rfio:/castor/cern.ch/user/p/pjanot/CMSSW310pre11/aod_QCDForPF_Fast_1.root',
       #'rfio:/castor/cern.ch/user/p/pjanot/CMSSW310pre11/aod_QCDForPF_Fast_2.root',
       #'rfio:/castor/cern.ch/user/p/pjanot/CMSSW310pre11/aod_QCDForPF_Fast_3.root',
       #'rfio:/castor/cern.ch/user/p/pjanot/CMSSW310pre11/aod_QCDForPF_Fast_4.root',
       #'rfio:/castor/cern.ch/user/p/pjanot/CMSSW310pre11/aod_QCDForPF_Fast_5.root',
       #'rfio:/castor/cern.ch/user/p/pjanot/CMSSW310pre11/aod_QCDForPF_Fast_6.root',
       #'rfio:/castor/cern.ch/user/p/pjanot/CMSSW310pre11/aod_QCDForPF_Fast_7.root',
       #'rfio:/castor/cern.ch/user/p/pjanot/CMSSW310pre11/aod_QCDForPF_Fast_8.root',
       #'rfio:/castor/cern.ch/user/p/pjanot/CMSSW310pre11/aod_QCDForPF_Fast_9.root',
       #'rfio:/castor/cern.ch/user/p/pjanot/CMSSW310pre11/aod_QCDForPF_Fast_10.root'
    # Full
       'rfio:/castor/cern.ch/user/p/pjanot/CMSSW310pre11/aod_QCDForPF_Full_001.root',
       'rfio:/castor/cern.ch/user/p/pjanot/CMSSW310pre11/aod_QCDForPF_Full_002.root',
       'rfio:/castor/cern.ch/user/p/pjanot/CMSSW310pre11/aod_QCDForPF_Full_003.root',
       #'file:aod.root'
       ),
    secondaryFileNames = cms.untracked.vstring(),
    noEventSort = cms.untracked.bool(True),
    duplicateCheckMode = cms.untracked.string('noDuplicateCheck')
    
    )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.load("Validation.RecoParticleFlow.pfJetBenchmark_cfi")
process.load("Validation.RecoParticleFlow.pfJetBenchmarkGeneric_cfi")
process.load("Validation.RecoParticleFlow.caloJetBenchmarkGeneric_cfi")
process.load("Validation.RecoParticleFlow.jptJetBenchmarkGeneric_cfi")
process.load("RecoJets.Configuration.GenJetParticles_cff")
process.load("RecoJets.Configuration.RecoGenJets_cff") 
process.load("RecoJets.Configuration.RecoPFJets_cff") 
process.load("PhysicsTools.HepMCCandAlgos.genParticles_cfi")
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.load("JetMETCorrections.Configuration.JetPlusTrackCorrections_cff")
process.load("JetMETCorrections.Configuration.ZSPJetCorrections152_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")

process.iterativeCone5PFJets.inputEtMin = 0.0
# Endcap
# process.pfJetBenchmarkGeneric.minEta = cms.double(1.6)
# process.caloJetBenchmarkGeneric.minEta = cms.double(1.6)
# process.jptJetBenchmarkGeneric.minEta = cms.double(1.6)
process.pfJetBenchmarkGeneric.maxEta = cms.double(5.0)
process.caloJetBenchmarkGeneric.maxEta = cms.double(5.0)
process.jptJetBenchmarkGeneric.maxEta = cms.double(5.0)

# should do a cloning
process.genParticlesForJets.ignoreParticleIDs.append(14)
process.genParticlesForJets.ignoreParticleIDs.append(12)
process.genParticlesForJets.ignoreParticleIDs.append(16)
process.genParticlesForJets.excludeResonances = False


process.pfJetBenchmarkGeneric.OutputFile = cms.untracked.string('JetBenchmarkGeneric.root')
process.caloJetBenchmarkGeneric.OutputFile = cms.untracked.string('JetBenchmarkGeneric.root')
process.jptJetBenchmarkGeneric.OutputFile = cms.untracked.string('JetBenchmarkGeneric.root')

process.p =cms.Path(
    process.genJetParticles+
    process.ak4GenJets+
    #process.iterativeCone5PFJets+
    process.pfJetBenchmarkGeneric+
    process.caloJetBenchmarkGeneric
    #process.ZSPJetCorrections+
    #process.JetPlusTrackCorrections+
    #process.jptJetBenchmarkGeneric
    )


process.schedule = cms.Schedule(process.p)



process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.options = cms.untracked.PSet(
    makeTriggerResults = cms.untracked.bool(True),
    wantSummary = cms.untracked.bool(True),
    Rethrow = cms.untracked.vstring('Unknown', 
        'ProductNotFound', 
        'DictionaryNotFound', 
        'InsertFailure', 
        'Configuration', 
        'LogicError', 
        'UnimplementedFeature', 
        'InvalidReference', 
        'NullPointerError', 
        'NoProductSpecified', 
        'EventTimeout', 
        'EventCorruption', 
        'ModuleFailure', 
        'ScheduleExecutionFailure', 
        'EventProcessorFailure', 
        'FileInPathError', 
        'FatalRootError', 
        'NotFound')
)

process.MessageLogger.cerr.FwkReport.reportEvery = 100
