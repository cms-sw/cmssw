import FWCore.ParameterSet.Config as cms

### genjet cleaning for improved matching in HI environment

from RecoHI.HiJetAlgos.HiGenCleaner_cff import *

iterativeCone5HiCleanedGenJets = heavyIonCleanedGenJets.clone( src = cms.InputTag('iterativeCone5HiGenJets'))
iterativeCone7HiCleanedGenJets = heavyIonCleanedGenJets.clone( src = cms.InputTag('iterativeCone7HiGenJets'))
ak4HiCleanedGenJets = heavyIonCleanedGenJets.clone( src = cms.InputTag('ak4HiGenJets'))
ak7HiCleanedGenJets = heavyIonCleanedGenJets.clone( src = cms.InputTag('ak7HiGenJets'))

### jet analyzer for two radii (0.5, 0.7) and three algorithms:
### iterative cone with PU, anti-kt with PU, anti-kt with fastjet PU


JetAnalyzerICPU5Calo = cms.EDAnalyzer("JetTester",
    JetType = cms.untracked.string('calo'),
    src = cms.InputTag("iterativeConePu5CaloJets"),
    srcGen = cms.InputTag("iterativeCone5HiCleanedGenJets"),
    primVertex     = cms.InputTag("hiSelectedVertex"),
    JetCorrections = cms.InputTag(""),
    recoJetPtThreshold = cms.double(40),
    matchGenPtThreshold                 = cms.double(20.0),
    RThreshold                     = cms.double(0.3)
)

JetAnalyzerICPU7Calo = cms.EDAnalyzer("JetTester",
    JetType = cms.untracked.string('calo'),
    src = cms.InputTag("iterativeConePu7CaloJets"),
    srcGen = cms.InputTag("iterativeCone7HiCleanedGenJets"),
    JetCorrections = cms.InputTag(""),
    recoJetPtThreshold = cms.double(40),
    matchGenPtThreshold                 = cms.double(20.0),
    RThreshold                     = cms.double(0.3)
)

JetAnalyzerAkPU5Calo = cms.EDAnalyzer("JetTester",
    JetType = cms.untracked.string('calo'),
    src = cms.InputTag("akPu5CaloJets"),
    srcGen = cms.InputTag("ak4HiCleanedGenJets"),
    primVertex     = cms.InputTag("hiSelectedVertex"),
    JetCorrections = cms.InputTag(""),
    recoJetPtThreshold = cms.double(40),
    matchGenPtThreshold                 = cms.double(20.0),
    RThreshold                     = cms.double(0.3)
)

JetAnalyzerAkPU7Calo = cms.EDAnalyzer("JetTester",
    JetType = cms.untracked.string('calo'),
    src = cms.InputTag("akPu7CaloJets"),
    srcGen = cms.InputTag("ak7HiCleanedGenJets"),
    primVertex     = cms.InputTag("hiSelectedVertex"),
    JetCorrections = cms.InputTag(""),
    recoJetPtThreshold = cms.double(40),
    matchGenPtThreshold                 = cms.double(20.0),
    RThreshold                     = cms.double(0.3)
)

JetAnalyzerAkFastPU5Calo = cms.EDAnalyzer("JetTester",
    JetType = cms.untracked.string('calo'),
    src = cms.InputTag("akFastPu5CaloJets"),
    srcGen = cms.InputTag("ak4HiCleanedGenJets"),
    primVertex     = cms.InputTag("hiSelectedVertex"),
    JetCorrections = cms.InputTag(""),
    recoJetPtThreshold = cms.double(40),
    matchGenPtThreshold                 = cms.double(20.0),
    RThreshold                     = cms.double(0.3)
)

JetAnalyzerAkFastPU7Calo = cms.EDAnalyzer("JetTester",
    JetType = cms.untracked.string('calo'),
    src = cms.InputTag("akFastPu7CaloJets"),
    srcGen = cms.InputTag("ak7HiCleanedGenJets"),
    primVertex     = cms.InputTag("hiSelectedVertex"),
    JetCorrections = cms.InputTag(""),
    recoJetPtThreshold = cms.double(40),
    matchGenPtThreshold                 = cms.double(20.0),
    RThreshold                     = cms.double(0.3)
)

hiJetValidation = cms.Sequence(
    (iterativeCone5HiCleanedGenJets * JetAnalyzerICPU5Calo)
    #+ (iterativeCone7HiCleanedGenJets * JetAnalyzerICPU7Calo)
    #+ (ak4HiCleanedGenJets * JetAnalyzerAkPU5Calo * JetAnalyzerAkFastPU5Calo
    #+ (ak7HiCleanedGenJets*JetAnalyzerAkPU7Calo *JetAnalyzerAkFastPU7Calo)
    )
