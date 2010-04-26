import FWCore.ParameterSet.Config as cms

### genjet cleaning for improved matching in HI environment

iterativeCone5HiCleanedGenJets = cms.EDProducer('HiGenJetCleaner',
  src    = cms.untracked.string('iterativeCone5HiGenJets'),
  deltaR = cms.untracked.double(0.25),
  ptCut  = cms.untracked.double(20),
  createNewCollection = cms.untracked.bool(True),
  fillDummyEntries = cms.untracked.bool(True)
)

iterativeCone7HiCleanedGenJets = cms.EDProducer('HiGenJetCleaner',
  src    = cms.untracked.string('iterativeCone7HiGenJets'),
  deltaR = cms.untracked.double(0.25),
  ptCut  = cms.untracked.double(20),
  createNewCollection = cms.untracked.bool(True),
  fillDummyEntries = cms.untracked.bool(True)
)

ak5HiCleanedGenJets = cms.EDProducer('HiGenJetCleaner',
  src    = cms.untracked.string('ak5HiGenJets'),
  deltaR = cms.untracked.double(0.25),
  ptCut  = cms.untracked.double(20),
  createNewCollection = cms.untracked.bool(True),
  fillDummyEntries = cms.untracked.bool(True)
)

ak7HiCleanedGenJets = cms.EDProducer('HiGenJetCleaner',
  src    = cms.untracked.string('ak7HiGenJets'),
  deltaR = cms.untracked.double(0.25),
  ptCut  = cms.untracked.double(20),
  createNewCollection = cms.untracked.bool(True),
  fillDummyEntries = cms.untracked.bool(True)
)

### jet analyzer for two radii (0.5, 0.7) and three algorithms:
### iterative cone with PU, anti-kt with PU, anti-kt with fastjet PU

JetAnalyzerICPU5Calo = cms.EDAnalyzer("CaloJetTester",
    src = cms.InputTag("iterativeConePu5CaloJets"),
    srcGen = cms.InputTag("iterativeCone5HiCleanedGenJets"),
    genEnergyFractionThreshold = cms.double(0.05),
    genPtThreshold = cms.double(1.0),
    RThreshold = cms.double(0.3),
    reverseEnergyFractionThreshold = cms.double(0.5)                                    
)

JetAnalyzerICPU7Calo = cms.EDAnalyzer("CaloJetTester",
    src = cms.InputTag("iterativeConePu7CaloJets"),
    srcGen = cms.InputTag("iterativeCone7HiCleanedGenJets"), 
    genEnergyFractionThreshold = cms.double(0.05),
    genPtThreshold = cms.double(1.0),
    RThreshold = cms.double(0.3),
    reverseEnergyFractionThreshold = cms.double(0.5)                                    
)

JetAnalyzerAkPU5Calo = cms.EDAnalyzer("CaloJetTester",
    src = cms.InputTag("akPu5CaloJets"),
    srcGen = cms.InputTag("ak5HiCleanedGenJets"),  
    genEnergyFractionThreshold = cms.double(0.05),
    genPtThreshold = cms.double(1.0),
    RThreshold = cms.double(0.3),
    reverseEnergyFractionThreshold = cms.double(0.5)                                    
)

JetAnalyzerAkPU7Calo = cms.EDAnalyzer("CaloJetTester",
    src = cms.InputTag("akPu7CaloJets"),
    srcGen = cms.InputTag("ak7HiCleanedGenJets"),       
    genEnergyFractionThreshold = cms.double(0.05),
    genPtThreshold = cms.double(1.0),
    RThreshold = cms.double(0.3),
    reverseEnergyFractionThreshold = cms.double(0.5)
)

JetAnalyzerAkFastPU5Calo = cms.EDAnalyzer("CaloJetTester",
    src = cms.InputTag("akFastPu5CaloJets"),
    srcGen = cms.InputTag("ak5HiCleanedGenJets"),  
    genEnergyFractionThreshold = cms.double(0.05),
    genPtThreshold = cms.double(1.0),
    RThreshold = cms.double(0.3),
    reverseEnergyFractionThreshold = cms.double(0.5)                                    
)

JetAnalyzerAkFastPU7Calo = cms.EDAnalyzer("CaloJetTester",
    src = cms.InputTag("akFastPu7CaloJets"),
    srcGen = cms.InputTag("ak7HiCleanedGenJets"),       
    genEnergyFractionThreshold = cms.double(0.05),
    genPtThreshold = cms.double(1.0),
    RThreshold = cms.double(0.3),
    reverseEnergyFractionThreshold = cms.double(0.5)
)

hiJetValidation = cms.Sequence(
    (iterativeCone5HiCleanedGenJets * JetAnalyzerICPU5Calo) 
    #+ (iterativeCone7HiCleanedGenJets * JetAnalyzerICPU7Calo)
    #+ (ak5HiCleanedGenJets * JetAnalyzerAkPU5Calo * JetAnalyzerAkFastPU5Calo
    #+ (ak7HiCleanedGenJets*JetAnalyzerAkPU7Calo *JetAnalyzerAkFastPU7Calo)
    )
