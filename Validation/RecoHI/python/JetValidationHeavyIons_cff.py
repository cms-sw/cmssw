import FWCore.ParameterSet.Config as cms

### genjet cleaning for improved matching in HI environment

from RecoHI.HiJetAlgos.HiGenCleaner_cff import *

iterativeCone5HiCleanedGenJets = heavyIonCleanedGenJets.clone( src = cms.InputTag('iterativeCone5HiGenJets'))
iterativeCone7HiCleanedGenJets = heavyIonCleanedGenJets.clone( src = cms.InputTag('iterativeCone7HiGenJets'))
ak3HiCleanedGenJets = heavyIonCleanedGenJets.clone( src = cms.InputTag('ak3HiGenJets'))
ak5HiCleanedGenJets = heavyIonCleanedGenJets.clone( src = cms.InputTag('ak5HiGenJets'))
ak7HiCleanedGenJets = heavyIonCleanedGenJets.clone( src = cms.InputTag('ak7HiGenJets'))

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

JetAnalyzerAkPU3Calo = cms.EDAnalyzer("CaloJetTester",
    src = cms.InputTag("akPu3CaloJets"),
    srcGen = cms.InputTag("ak3HiCleanedGenJets"),       
    genEnergyFractionThreshold = cms.double(0.05),
    genPtThreshold = cms.double(1.0),
    RThreshold = cms.double(0.3),
    reverseEnergyFractionThreshold = cms.double(0.5)
)

JetAnalyzerAkPU5PF = cms.EDAnalyzer("CaloJetTester",
                                    src = cms.InputTag("akPu5PFJets"),
                                    srcGen = cms.InputTag("ak5HiCleanedGenJets"),
                                    genEnergyFractionThreshold = cms.double(0.05),
                                    genPtThreshold = cms.double(1.0),
                                    RThreshold = cms.double(0.3),
                                    reverseEnergyFractionThreshold = cms.double(0.5)
                                    )

JetAnalyzerAkPU3PF = cms.EDAnalyzer("CaloJetTester",
                                    src = cms.InputTag("akPu3PFJets"),
                                    srcGen = cms.InputTag("ak3HiCleanedGenJets"),
                                    genEnergyFractionThreshold = cms.double(0.05),
                                    genPtThreshold = cms.double(1.0),
                                    RThreshold = cms.double(0.3),
                                    reverseEnergyFractionThreshold = cms.double(0.5)
                                    )


hiJetValidation = cms.Sequence(
    ak3HiCleanedGenJets * ak5HiCleanedGenJets
    * JetAnalyzerAkPU5Calo
    * JetAnalyzerAkPU3PF * JetAnalyzerAkPU5PF     
    )
