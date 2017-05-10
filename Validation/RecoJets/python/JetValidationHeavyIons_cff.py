import FWCore.ParameterSet.Config as cms

### genjet cleaning for improved matching in HI environment

from RecoHI.HiJetAlgos.HiGenCleaner_cff import *

iterativeCone5HiCleanedGenJets = heavyIonCleanedGenJets.clone( src = cms.InputTag('iterativeCone5HiGenJets'))
iterativeCone7HiCleanedGenJets = heavyIonCleanedGenJets.clone( src = cms.InputTag('iterativeCone7HiGenJets'))
ak3HiCleanedGenJets = heavyIonCleanedGenJets.clone( src = cms.InputTag('ak3HiGenJets'))
ak5HiCleanedGenJets = heavyIonCleanedGenJets.clone( src = cms.InputTag('ak5HiGenJets'))
ak7HiCleanedGenJets = heavyIonCleanedGenJets.clone( src = cms.InputTag('ak7HiGenJets'))

### jet analyzer for several radii
### iterative cone with PU, anti-kt with PU, anti-kt with fastjet PU, anti-kt with Vs 

JetAnalyzerICPU5Calo = cms.EDAnalyzer("JetTester_HeavyIons",
    JetType = cms.untracked.string('calo'),
    OutputFile = cms.untracked.string('JetTester.root'),
    src = cms.InputTag("iterativeConePu5CaloJets"),
    srcGen = cms.InputTag("iterativeCone5HiCleanedGenJets"),
    srcRho = cms.InputTag("iterativeConePu5CaloJets","rho"), 
    JetCorrections = cms.string(""),
    recoJetPtThreshold = cms.double(10),
    genEnergyFractionThreshold = cms.double(0.05),
    genPtThreshold = cms.double(1.0),
    matchGenPtThreshold                 = cms.double(20.0),
    RThreshold = cms.double(0.3),
    reverseEnergyFractionThreshold = cms.double(0.5)                                    
    )

JetAnalyzerICPU7Calo = cms.EDAnalyzer("JetTester_HeavyIons",
    JetType = cms.untracked.string('calo'),
    OutputFile = cms.untracked.string('JetTester.root'),
    src = cms.InputTag("iterativeConePu7CaloJets"),
    srcGen = cms.InputTag("iterativeCone7HiCleanedGenJets"),
    srcRho = cms.InputTag("iterativeConePu7CaloJets","rho"),	 
    JetCorrections = cms.string(""),
    recoJetPtThreshold = cms.double(10),
    genEnergyFractionThreshold = cms.double(0.05),
    genPtThreshold = cms.double(1.0),
    matchGenPtThreshold                 = cms.double(20.0),
    RThreshold = cms.double(0.3),
    reverseEnergyFractionThreshold = cms.double(0.5)                                    
    )

JetAnalyzerAkPU5Calo = cms.EDAnalyzer("JetTester_HeavyIons",
    JetType = cms.untracked.string('calo'),
    OutputFile = cms.untracked.string('JetTester.root'),
    src = cms.InputTag("akPu5CaloJets"),
    srcGen = cms.InputTag("ak5HiCleanedGenJets"), 
    srcRho = cms.InputTag("akPu5CaloJets","rho"),	 
    JetCorrections = cms.string(""),
    recoJetPtThreshold = cms.double(10),        
    genEnergyFractionThreshold = cms.double(0.05),
    genPtThreshold = cms.double(1.0),
    matchGenPtThreshold                 = cms.double(20.0),
    RThreshold = cms.double(0.3),
    reverseEnergyFractionThreshold = cms.double(0.5)                                    
    )

JetAnalyzerAkPU3Calo = cms.EDAnalyzer("JetTester_HeavyIons",
    JetType = cms.untracked.string('calo'),
    OutputFile = cms.untracked.string('JetTester.root'),
    src = cms.InputTag("akPu3CaloJets"),
    srcGen = cms.InputTag("ak3HiCleanedGenJets"),       
    srcRho = cms.InputTag("akPu3CaloJets","rho"),           
    JetCorrections = cms.string(""),
    recoJetPtThreshold = cms.double(10),        
    genEnergyFractionThreshold = cms.double(0.05),
    genPtThreshold = cms.double(1.0),
    matchGenPtThreshold                 = cms.double(20.0),
    RThreshold = cms.double(0.3),
    reverseEnergyFractionThreshold = cms.double(0.5)
    )

JetAnalyzerAkPU5PF = cms.EDAnalyzer("JetTester_HeavyIons",
    JetType = cms.untracked.string('pf'),
    OutputFile = cms.untracked.string('JetTester.root'),
    src = cms.InputTag("akPu5PFJets"),
    srcGen = cms.InputTag("ak5HiCleanedGenJets"),
    srcRho = cms.InputTag("akPu5PFJets","rho"),
    JetCorrections = cms.string(""),
    recoJetPtThreshold = cms.double(10),        
    genEnergyFractionThreshold = cms.double(0.05),
    genPtThreshold = cms.double(1.0),
    matchGenPtThreshold                 = cms.double(20.0),
    RThreshold = cms.double(0.3),
    reverseEnergyFractionThreshold = cms.double(0.5)
    )

JetAnalyzerAkPU3PF = cms.EDAnalyzer("JetTester_HeavyIons",
    JetType = cms.untracked.string('pf'),
    OutputFile = cms.untracked.string('JetTester.root'),
    src = cms.InputTag("akPu3PFJets"),
    srcGen = cms.InputTag("ak3HiCleanedGenJets"),
    srcRho = cms.InputTag("akPu3PFJets","rho"),
    JetCorrections = cms.string(""),
    recoJetPtThreshold = cms.double(10),        
    genEnergyFractionThreshold = cms.double(0.05),
    genPtThreshold = cms.double(1.0),
    matchGenPtThreshold                 = cms.double(20.0),
    RThreshold = cms.double(0.3),
    reverseEnergyFractionThreshold = cms.double(0.5)
    )


JetAnalyzerAkVs2Calo = cms.EDAnalyzer("JetTester_HeavyIons",
    JetType = cms.untracked.string('calo'),
    OutputFile = cms.untracked.string('JetTester.root'),
    src = cms.InputTag("akVs2CaloJets"),
    srcGen = cms.InputTag("ak2HiCleanedGenJets"),
    srcRho = cms.InputTag("akVs2CaloJets","rho"),
    JetCorrections = cms.string(""),
    recoJetPtThreshold = cms.double(10),        
    genEnergyFractionThreshold = cms.double(0.05),
    genPtThreshold = cms.double(1.0),
    matchGenPtThreshold                 = cms.double(20.0),
    RThreshold = cms.double(0.3),
    reverseEnergyFractionThreshold = cms.double(0.5)
    )

JetAnalyzerAkVs3Calo = cms.EDAnalyzer("JetTester_HeavyIons",
    JetType = cms.untracked.string('calo'),
    OutputFile = cms.untracked.string('JetTester.root'),
    src = cms.InputTag("akVs3CaloJets"),
    srcGen = cms.InputTag("ak3HiCleanedGenJets"),
    srcRho = cms.InputTag("akVs3CaloJets","rho"),
    JetCorrections = cms.string(""),
    recoJetPtThreshold = cms.double(10),        
    genEnergyFractionThreshold = cms.double(0.05),
    genPtThreshold = cms.double(1.0),
    matchGenPtThreshold                 = cms.double(20.0),
    RThreshold = cms.double(0.3),
    reverseEnergyFractionThreshold = cms.double(0.5)
    )

JetAnalyzerAkVs4Calo = cms.EDAnalyzer("JetTester_HeavyIons",
    JetType = cms.untracked.string('calo'),
    OutputFile = cms.untracked.string('JetTester.root'),
    src = cms.InputTag("akVs4CaloJets"),
    srcGen = cms.InputTag("ak4HiCleanedGenJets"),
    srcRho = cms.InputTag("akVs4CaloJets","rho"),
    JetCorrections = cms.string(""),
    recoJetPtThreshold = cms.double(10),        
    genEnergyFractionThreshold = cms.double(0.05),
    genPtThreshold = cms.double(1.0),
    matchGenPtThreshold                 = cms.double(20.0),
    RThreshold = cms.double(0.3),
    reverseEnergyFractionThreshold = cms.double(0.5)
    )

JetAnalyzerAkVs5Calo = cms.EDAnalyzer("JetTester_HeavyIons",
    JetType = cms.untracked.string('calo'),
    OutputFile = cms.untracked.string('JetTester.root'),
    src = cms.InputTag("akVs5CaloJets"),
    srcGen = cms.InputTag("ak5HiCleanedGenJets"),
    srcRho = cms.InputTag("akVs5CaloJets","rho"),
    JetCorrections = cms.string(""),
    recoJetPtThreshold = cms.double(10),        
    genEnergyFractionThreshold = cms.double(0.05),
    genPtThreshold = cms.double(1.0),
    matchGenPtThreshold                 = cms.double(20.0),
    RThreshold = cms.double(0.3),
    reverseEnergyFractionThreshold = cms.double(0.5)
    )

JetAnalyzerAkVs6Calo = cms.EDAnalyzer("JetTester_HeavyIons",
    JetType = cms.untracked.string('calo'),
    OutputFile = cms.untracked.string('JetTester.root'),
    src = cms.InputTag("akVs6CaloJets"),
    srcGen = cms.InputTag("ak6HiCleanedGenJets"),
    srcRho = cms.InputTag("akVs6CaloJets","rho"),
    JetCorrections = cms.string(""),
    recoJetPtThreshold = cms.double(10),        
    genEnergyFractionThreshold = cms.double(0.05),
    genPtThreshold = cms.double(1.0),
    matchGenPtThreshold                 = cms.double(20.0),
    RThreshold = cms.double(0.3),
    reverseEnergyFractionThreshold = cms.double(0.5)
    )


JetAnalyzerAkVs7Calo = cms.EDAnalyzer("JetTester_HeavyIons",
    JetType = cms.untracked.string('calo'),
    OutputFile = cms.untracked.string('JetTester.root'),
    src = cms.InputTag("akVs7CaloJets"),
    srcGen = cms.InputTag("ak7HiCleanedGenJets"),
    srcRho = cms.InputTag("akVs7CaloJets","rho"),
    JetCorrections = cms.string(""),
    recoJetPtThreshold = cms.double(10),        
    genEnergyFractionThreshold = cms.double(0.05),
    genPtThreshold = cms.double(1.0),
    matchGenPtThreshold                 = cms.double(20.0),
    RThreshold = cms.double(0.3),
    reverseEnergyFractionThreshold = cms.double(0.5)
    )

JetAnalyzerAkVs2PF = cms.EDAnalyzer("JetTester_HeavyIons",
    JetType = cms.untracked.string('pf'),
    OutputFile = cms.untracked.string('JetTester.root'),
    src = cms.InputTag("akVs2PFJets"),
    srcGen = cms.InputTag("ak2HiCleanedGenJets"),
    srcRho = cms.InputTag("akVs2PFJets","rho"),
    JetCorrections = cms.string(""),
    recoJetPtThreshold = cms.double(10),        
    genEnergyFractionThreshold = cms.double(0.05),
    genPtThreshold = cms.double(1.0),
    matchGenPtThreshold                 = cms.double(20.0),
    RThreshold = cms.double(0.3),
    reverseEnergyFractionThreshold = cms.double(0.5)
    )

JetAnalyzerAkVs3PF = cms.EDAnalyzer("JetTester_HeavyIons",
    JetType = cms.untracked.string('pf'),
    OutputFile = cms.untracked.string('JetTester.root'),
    src = cms.InputTag("akVs3PFJets"),
    srcGen = cms.InputTag("ak3HiCleanedGenJets"),
    srcRho = cms.InputTag("akVs3PFJets","rho"),
    JetCorrections = cms.string(""),
    recoJetPtThreshold = cms.double(10),        
    genEnergyFractionThreshold = cms.double(0.05),
    genPtThreshold = cms.double(1.0),
    matchGenPtThreshold                 = cms.double(20.0),
    RThreshold = cms.double(0.3),
    reverseEnergyFractionThreshold = cms.double(0.5)
    )

JetAnalyzerAkVs4PF = cms.EDAnalyzer("JetTester_HeavyIons",
    JetType = cms.untracked.string('pf'),
    OutputFile = cms.untracked.string('JetTester.root'),
    src = cms.InputTag("akVs4PFJets"),
    srcGen = cms.InputTag("ak4HiCleanedGenJets"),
    srcRho = cms.InputTag("akVs4PFJets","rho"),
    JetCorrections = cms.string(""),
    recoJetPtThreshold = cms.double(10),        
    genEnergyFractionThreshold = cms.double(0.05),
    genPtThreshold = cms.double(1.0),
    matchGenPtThreshold                 = cms.double(20.0),
    RThreshold = cms.double(0.3),
    reverseEnergyFractionThreshold = cms.double(0.5)
    )

JetAnalyzerAkVs5PF = cms.EDAnalyzer("JetTester_HeavyIons",
    JetType = cms.untracked.string('pf'),
    OutputFile = cms.untracked.string('JetTester.root'),
    src = cms.InputTag("akVs5PFJets"),
    srcGen = cms.InputTag("ak5HiCleanedGenJets"),
    srcRho = cms.InputTag("akVs5PFJets","rho"),
    JetCorrections = cms.string(""),
    recoJetPtThreshold = cms.double(10),        
    genEnergyFractionThreshold = cms.double(0.05),
    genPtThreshold = cms.double(1.0),
    matchGenPtThreshold                 = cms.double(20.0),
    RThreshold = cms.double(0.3),
    reverseEnergyFractionThreshold = cms.double(0.5)
    )

JetAnalyzerAkVs6PF = cms.EDAnalyzer("JetTester_HeavyIons",
    JetType = cms.untracked.string('pf'),
    OutputFile = cms.untracked.string('JetTester.root'),
    src = cms.InputTag("akVs6PFJets"),
    srcGen = cms.InputTag("ak6HiCleanedGenJets"),
    srcRho = cms.InputTag("akVs6PFJets","rho"),
    JetCorrections = cms.string(""),
    recoJetPtThreshold = cms.double(10),        
    genEnergyFractionThreshold = cms.double(0.05),
    genPtThreshold = cms.double(1.0),
    matchGenPtThreshold                 = cms.double(20.0),
    RThreshold = cms.double(0.3),
    reverseEnergyFractionThreshold = cms.double(0.5)
    )

JetAnalyzerAkVs7PF = cms.EDAnalyzer("JetTester_HeavyIons",
    JetType = cms.untracked.string('pf'),
    OutputFile = cms.untracked.string('JetTester.root'),
    src = cms.InputTag("akVs7PFJets"),
    srcGen = cms.InputTag("ak7HiCleanedGenJets"),
    srcRho = cms.InputTag("akVs7PFJets","rho"),
    JetCorrections = cms.string(""),
    recoJetPtThreshold = cms.double(10),        
    genEnergyFractionThreshold = cms.double(0.05),
    genPtThreshold = cms.double(1.0),
    matchGenPtThreshold                 = cms.double(20.0),
    RThreshold = cms.double(0.3),
    reverseEnergyFractionThreshold = cms.double(0.5)
    )				    			    

hiJetValidation = cms.Sequence(
    ak3HiCleanedGenJets
#    * ak5HiCleanedGenJets 
#* ak7HiCleanedGenJets
    * JetAnalyzerICPU5Calo
#* JetAnalyzerICPU7Calo
    * JetAnalyzerAkPU3Calo
#* JetAnalyzerAkPU5Calo
    * JetAnalyzerAkPU3PF
#* JetAnalyzerAkPU5PF

#* JetAnalyzerAkVs2Calo	   
    * JetAnalyzerAkVs3Calo	   
#* JetAnalyzerAkVs4Calo	   
#* JetAnalyzerAkVs5Calo	   
   #* JetAnalyzerAkVs6Calo
   #* JetAnalyzerAkVs7Calo

   #* JetAnalyzerAkVs2PF
   * JetAnalyzerAkVs3PF
   #* JetAnalyzerAkVs4PF	   
   #* JetAnalyzerAkVs5PF
   #* JetAnalyzerAkVs6PF	   
   #* JetAnalyzerAkVs7PF

    )
