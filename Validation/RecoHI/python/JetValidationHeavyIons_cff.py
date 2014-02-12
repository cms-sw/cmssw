import FWCore.ParameterSet.Config as cms

### genjet cleaning for improved matching in HI environment

from RecoHI.HiJetAlgos.HiGenCleaner_cff import *

iterativeCone5HiCleanedGenJets = heavyIonCleanedGenJets.clone( src = cms.InputTag('iterativeCone5HiGenJets'))
iterativeCone7HiCleanedGenJets = heavyIonCleanedGenJets.clone( src = cms.InputTag('iterativeCone7HiGenJets'))
ak5HiCleanedGenJets = heavyIonCleanedGenJets.clone( src = cms.InputTag('ak5HiGenJets'))
ak7HiCleanedGenJets = heavyIonCleanedGenJets.clone( src = cms.InputTag('ak7HiGenJets'))

### jet analyzer for two radii (0.5, 0.7) and three algorithms:
### iterative cone with PU, anti-kt with PU, anti-kt with fastjet PU

JetAnalyzerICPU5Calo = cms.EDAnalyzer("CaloJetTesterUnCorr_HeavyIons",
    src = cms.InputTag("iterativeConePu5CaloJets"),
    srcGen = cms.InputTag("iterativeCone5HiCleanedGenJets"),
    srcRho = cms.InputTag("iterativeConePu5CaloJets","rho"),
    genEnergyFractionThreshold = cms.double(0.05),
    genPtThreshold = cms.double(1.0),
    RThreshold = cms.double(0.3),
    reverseEnergyFractionThreshold = cms.double(0.5)                                    
)

JetAnalyzerICPU7Calo = cms.EDAnalyzer("CaloJetTesterUnCorr_HeavyIons",
    src = cms.InputTag("iterativeConePu7CaloJets"),
    srcGen = cms.InputTag("iterativeCone7HiCleanedGenJets"), 
    srcRho = cms.InputTag("iterativeConePu7CaloJets","rho"),
    genEnergyFractionThreshold = cms.double(0.05),
    genPtThreshold = cms.double(1.0),
    RThreshold = cms.double(0.3),
    reverseEnergyFractionThreshold = cms.double(0.5)                                    
)

JetAnalyzerAkPU5Calo = cms.EDAnalyzer("CaloJetTesterUnCorr_HeavyIons",
    src = cms.InputTag("akPu5CaloJets"),
    srcGen = cms.InputTag("ak5HiCleanedGenJets"),  
    srcRho = cms.InputTag("akPu5CaloJets","rho"),
    genEnergyFractionThreshold = cms.double(0.05),
    genPtThreshold = cms.double(1.0),
    RThreshold = cms.double(0.3),
    reverseEnergyFractionThreshold = cms.double(0.5)                                    
)

JetAnalyzerAkPU7Calo = cms.EDAnalyzer("CaloJetTesterUnCorr_HeavyIons",
    src = cms.InputTag("akPu7CaloJets"),
    srcGen = cms.InputTag("ak7HiCleanedGenJets"),       
    srcRho = cms.InputTag("akPu7CaloJets","rho"),
    genEnergyFractionThreshold = cms.double(0.05),
    genPtThreshold = cms.double(1.0),
    RThreshold = cms.double(0.3),
    reverseEnergyFractionThreshold = cms.double(0.5)
)

JetAnalyzerAkFastPU5Calo = cms.EDAnalyzer("CaloJetTesterUnCorr_HeavyIons",
    src = cms.InputTag("akFastPu5CaloJets"),
    srcGen = cms.InputTag("ak5HiCleanedGenJets"),  
    srcRho = cms.InputTag("akFastPu5CaloJets","rho"),
    genEnergyFractionThreshold = cms.double(0.05),
    genPtThreshold = cms.double(1.0),
    RThreshold = cms.double(0.3),
    reverseEnergyFractionThreshold = cms.double(0.5)                                    
)

JetAnalyzerAkFastPU7Calo = cms.EDAnalyzer("CaloJetTesterUnCorr_HeavyIons",
    src = cms.InputTag("akFastPu7CaloJets"),
    srcGen = cms.InputTag("ak7HiCleanedGenJets"),
    srcRho = cms.InputTag("akFastPu5CaloJets","rho"),
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
