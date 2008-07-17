import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("DQMServices.Core.DQM_cfg")

process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("RecoJets.Configuration.GenJetParticles_cff")

process.load("RecoJets.Configuration.RecoGenJets_cff")

process.load("RecoJets.Configuration.RecoPFJets_cff")

process.load("PhysicsTools.HepMCCandAlgos.genParticles_cfi")

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:MyFirstFamosFile.root')
)

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
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.pfAnalyzer = cms.EDAnalyzer("PFBenchmarkAnalyzer",
    OutputFile = cms.untracked.string('PFJetTester_data.root'),
    InputTruthLabel = cms.string('iterativeCone5GenJets'),
    maxEta = cms.double(2.0),
    recPt = cms.double(20.0),
    PlotAgainstRecoQuantities = cms.bool(True),
    BenchmarkLabel = cms.string('ParticleFlow'),
    InputRecoLabel = cms.string('iterativeCone5PFJets')
)

process.pfTester = cms.EDFilter("PFTester",
    InputPFlowLabel = cms.string('particleFlow'),
    OutputFile = cms.untracked.string('PFTester_data.root')
)

process.pfJetAnalyzer = cms.EDFilter("JetAna",
    OutputFile = cms.untracked.string('PFJetTester_data.root'),
    InputTruthLabel = cms.string('iterativeCone5GenJets'),
    maxEta = cms.double(3.0),
    recPt = cms.double(20.0),
    PlotAgainstRecoQuantities = cms.bool(True),
    pfjBenchmarkDebug = cms.bool(False),
    deltaRMax = cms.double(0.1),
    BenchmarkLabel = cms.string('ParticleFlow'),
    InputRecoLabel = cms.string('iterativeCone5PFJets')
)

process.o = cms.OutputModule("PoolOutputModule",
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('p')
    ),
    fileName = cms.untracked.string('test.root')
)

process.p = cms.Path(process.genParticles*process.genJetParticles*process.recoGenJets*process.pfAnalyzer*process.pfJetAnalyzer)
process.schedule = cms.Schedule(process.p)

process.MessageLogger.cerr.FwkReport.reportEvery = 100
process.genParticlesForJets.ignoreParticleIDs.append(14)
process.genParticlesForJets.ignoreParticleIDs.append(12)
process.genParticlesForJets.ignoreParticleIDs.append(16)
process.genParticlesForJets.excludeResonances = False


