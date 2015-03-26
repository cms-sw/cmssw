import FWCore.ParameterSet.Config as cms

### genjet cleaning for improved matching in HI environment

from RecoHI.HiJetAlgos.HiGenCleaner_cff import *

iterativeCone5HiCleanedGenJets = heavyIonCleanedGenJets.clone( src = cms.InputTag('iterativeCone5HiGenJets'))
#iterativeCone7HiCleanedGenJets = heavyIonCleanedGenJets.clone( src = cms.InputTag('iterativeCone7HiGenJets'))
ak2HiCleanedGenJets = heavyIonCleanedGenJets.clone( src = cms.InputTag('ak2HiGenJets'))
ak3HiCleanedGenJets = heavyIonCleanedGenJets.clone( src = cms.InputTag('ak3HiGenJets'))
ak4HiCleanedGenJets = heavyIonCleanedGenJets.clone( src = cms.InputTag('ak4HiGenJets'))
ak5HiCleanedGenJets = heavyIonCleanedGenJets.clone( src = cms.InputTag('ak5HiGenJets'))

ak7HiCleanedGenJets = heavyIonCleanedGenJets.clone( src = cms.InputTag('ak7HiGenJets'))

### jet analyzer for several radii
### iterative cone with PU, anti-kt with PU, anti-kt with fastjet PU, anti-kt with Vs 


JetAnalyzerICPU5Calo = cms.EDAnalyzer("JetTester_HeavyIons",
                                      JetType = cms.untracked.string('calo'),
                                      UEAlgo = cms.untracked.string('Pu'),
                                      OutputFile = cms.untracked.string(''),
                                      src = cms.InputTag("iterativeConePu5CaloJets"),
                                      srcGen = cms.InputTag("iterativeCone5HiCleanedGenJets"),
                                      PFcands = cms.InputTag("particleFlowTmp"),
                                      Background = cms.InputTag("voronoiBackgroundCalo"),
                                      #srcRho = cms.InputTag("iterativeConePu5CaloJets","rho"),
                                      centrality = cms.InputTag("hiCentrality"),
                                      JetCorrections = cms.string(""),
                                      recoJetPtThreshold = cms.double(10),
                                      genEnergyFractionThreshold = cms.double(0.05),
                                      genPtThreshold = cms.double(1.0),
                                      matchGenPtThreshold                 = cms.double(20.0),
                                      RThreshold = cms.double(0.3),
                                      reverseEnergyFractionThreshold = cms.double(0.5)                                    
)
'''
JetAnalyzerICPU7Calo = cms.EDAnalyzer("JetTester_HeavyIons",
                                      JetType = cms.untracked.string('calo'),                       
                                      UEAlgo = cms.untracked.string('Pu'),                                    
                                      OutputFile = cms.untracked.string(''),
                                      src = cms.InputTag("iterativeConePu7CaloJets"),
                                      srcGen = cms.InputTag("iterativeCone7HiCleanedGenJets"),
                                      #srcRho = cms.InputTag("iterativeConePu7CaloJets","rho"), 
                                      JetCorrections = cms.string(""),
                                      recoJetPtThreshold = cms.double(10),
                                      genEnergyFractionThreshold = cms.double(0.05),
                                      genPtThreshold = cms.double(1.0),
                                      matchGenPtThreshold                 = cms.double(20.0),
                                      RThreshold = cms.double(0.3),
                                      reverseEnergyFractionThreshold = cms.double(0.5)                                    
)
'''
JetAnalyzerAkPU3Calo = cms.EDAnalyzer("JetTester_HeavyIons",
                                      JetType = cms.untracked.string('calo'),
                                      UEAlgo = cms.untracked.string('Pu'),
                                      OutputFile = cms.untracked.string(''),
                                      src = cms.InputTag("akPu3CaloJets"),
                                      srcGen = cms.InputTag("ak3HiCleanedGenJets"),       
                                      PFcands = cms.InputTag("particleFlowTmp"),
                                      Background = cms.InputTag("voronoiBackgroundCalo"),
                                      #srcRho = cms.InputTag("iterativeConePu5CaloJets","rho"),
                                      centrality = cms.InputTag("hiCentrality"),
                                      JetCorrections = cms.string(""),
                                      recoJetPtThreshold = cms.double(10),        
                                      genEnergyFractionThreshold = cms.double(0.05),
                                      genPtThreshold = cms.double(1.0),
                                      matchGenPtThreshold                 = cms.double(20.0),
                                      RThreshold = cms.double(0.3),
                                      reverseEnergyFractionThreshold = cms.double(0.5)
)

JetAnalyzerAkPU4Calo = cms.EDAnalyzer("JetTester_HeavyIons",
                                      JetType = cms.untracked.string('calo'),
                                      UEAlgo = cms.untracked.string('Pu'),
                                      OutputFile = cms.untracked.string(''),
                                      src = cms.InputTag("akPu4CaloJets"),
                                      srcGen = cms.InputTag("ak4HiCleanedGenJets"), 
                                      PFcands = cms.InputTag("particleFlowTmp"),
                                      Background = cms.InputTag("voronoiBackgroundCalo"),
                                      #srcRho = cms.InputTag("iterativeConePu5CaloJets","rho"),
                                      centrality = cms.InputTag("hiCentrality"),
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
                                      UEAlgo = cms.untracked.string('Pu'),
                                      OutputFile = cms.untracked.string(''),
                                      src = cms.InputTag("akPu5CaloJets"),
                                      srcGen = cms.InputTag("ak5HiCleanedGenJets"),       
                                      PFcands = cms.InputTag("particleFlowTmp"),
                                      Background = cms.InputTag("voronoiBackgroundCalo"),
                                      #srcRho = cms.InputTag("iterativeConePu5CaloJets","rho"),
                                      centrality = cms.InputTag("hiCentrality"),
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
                                    UEAlgo = cms.untracked.string('Pu'),
                                    OutputFile = cms.untracked.string(''),
                                    src = cms.InputTag("akPu3PFJets"),
                                    srcGen = cms.InputTag("ak3HiCleanedGenJets"),
                                    PFcands = cms.InputTag("particleFlowTmp"),
                                    Background = cms.InputTag("voronoiBackgroundPF"),
                                    #srcRho = cms.InputTag("iterativeConePu5CaloJets","rho"),
                                    centrality = cms.InputTag("hiCentrality"),
                                    JetCorrections = cms.string(""),
                                    recoJetPtThreshold = cms.double(10),        
                                    genEnergyFractionThreshold = cms.double(0.05),
                                    genPtThreshold = cms.double(1.0),
                                    matchGenPtThreshold                 = cms.double(20.0),
                                    RThreshold = cms.double(0.3),
                                    reverseEnergyFractionThreshold = cms.double(0.5)
)

JetAnalyzerAkPU4PF = cms.EDAnalyzer("JetTester_HeavyIons",
                                    JetType = cms.untracked.string('pf'),
                                    UEAlgo = cms.untracked.string('Pu'),
                                    OutputFile = cms.untracked.string(''),
                                    src = cms.InputTag("akPu4PFJets"),
                                    srcGen = cms.InputTag("ak4HiCleanedGenJets"),
                                    PFcands = cms.InputTag("particleFlowTmp"),
                                    Background = cms.InputTag("voronoiBackgroundPF"),
                                    #srcRho = cms.InputTag("iterativeConePu5CaloJets","rho"),
                                    centrality = cms.InputTag("hiCentrality"),
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
                                    UEAlgo = cms.untracked.string('Pu'),
                                    OutputFile = cms.untracked.string(''),
                                    src = cms.InputTag("akPu5PFJets"),
                                    srcGen = cms.InputTag("ak5HiCleanedGenJets"),
                                    PFcands = cms.InputTag("particleFlowTmp"),
                                    Background = cms.InputTag("voronoiBackgroundPF"),
                                    #srcRho = cms.InputTag("iterativeConePu5CaloJets","rho"),
                                    centrality = cms.InputTag("hiCentrality"),
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
                                      UEAlgo = cms.untracked.string('Vs'),
                                      OutputFile = cms.untracked.string(''),
                                      src = cms.InputTag("akVs2CaloJets"),
                                      srcGen = cms.InputTag("ak2HiCleanedGenJets"),
                                      PFcands = cms.InputTag("particleFlowTmp"),
                                      #Cands = cms.InputTag("caloTowers"),
                                      Background = cms.InputTag("voronoiBackgroundCalo"),
                                      #srcRho = cms.InputTag("iterativeConePu5CaloJets","rho"),
                                      centrality = cms.InputTag("hiCentrality"),
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
                                      UEAlgo = cms.untracked.string('Vs'),
                                      OutputFile = cms.untracked.string(''),
                                      src = cms.InputTag("akVs3CaloJets"),
                                      srcGen = cms.InputTag("ak3HiCleanedGenJets"),
                                      PFcands = cms.InputTag("particleFlowTmp"),
                                      Background = cms.InputTag("voronoiBackgroundCalo"),
                                      #srcRho = cms.InputTag("iterativeConePu5CaloJets","rho"),
                                      centrality = cms.InputTag("hiCentrality"),
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
                                      UEAlgo = cms.untracked.string('Vs'),
                                      OutputFile = cms.untracked.string(''),
                                      src = cms.InputTag("akVs4CaloJets"),
                                      srcGen = cms.InputTag("ak4HiCleanedGenJets"),
                                      PFcands = cms.InputTag("particleFlowTmp"),
                                      Background = cms.InputTag("voronoiBackgroundCalo"),
                                      #srcRho = cms.InputTag("iterativeConePu5CaloJets","rho"),
                                      centrality = cms.InputTag("hiCentrality"),
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
                                      UEAlgo = cms.untracked.string('Vs'),
                                      OutputFile = cms.untracked.string(''),
                                      src = cms.InputTag("akVs5CaloJets"),
                                      srcGen = cms.InputTag("ak5HiCleanedGenJets"),
                                      PFcands = cms.InputTag("particleFlowTmp"),
                                      Background = cms.InputTag("voronoiBackgroundCalo"),
                                      #srcRho = cms.InputTag("iterativeConePu5CaloJets","rho"),
                                      centrality = cms.InputTag("hiCentrality"),
                                      JetCorrections = cms.string(""),
                                      recoJetPtThreshold = cms.double(10),        
                                      genEnergyFractionThreshold = cms.double(0.05),
                                      genPtThreshold = cms.double(1.0),
                                      matchGenPtThreshold                 = cms.double(20.0),
                                      RThreshold = cms.double(0.3),
                                      reverseEnergyFractionThreshold = cms.double(0.5)
)
'''
JetAnalyzerAkVs6Calo = cms.EDAnalyzer("JetTester_HeavyIons",
                                      JetType = cms.untracked.string('calo'),
                                      UEAlgo = cms.untracked.string('Vs'),
                                      OutputFile = cms.untracked.string(''),
                                      src = cms.InputTag("akVs6CaloJets"),
                                      srcGen = cms.InputTag("ak6HiCleanedGenJets"),
                                      PFcands = cms.InputTag("particleFlowTmp"),
                                      Background = cms.InputTag("voronoiBackgroundPF"),
                                      #srcRho = cms.InputTag("iterativeConePu5CaloJets","rho"),
                                      centrality = cms.InputTag("hiCentrality"),
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
                                      UEAlgo = cms.untracked.string('Vs'),
                                      OutputFile = cms.untracked.string(''),
                                      src = cms.InputTag("akVs7CaloJets"),
                                      srcGen = cms.InputTag("ak7HiCleanedGenJets"),
                                      PFcands = cms.InputTag("particleFlowTmp"),
                                      Background = cms.InputTag("voronoiBackgroundPF"),
                                      #srcRho = cms.InputTag("iterativeConePu5CaloJets","rho"),
                                      centrality = cms.InputTag("hiCentrality"),
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
                                    UEAlgo = cms.untracked.string('Vs'),
                                    OutputFile = cms.untracked.string(''),
                                    src = cms.InputTag("akVs2PFJets"),
                                    srcGen = cms.InputTag("ak2HiCleanedGenJets"),
                                      PFcands = cms.InputTag("particleFlowTmp"),
                                      Background = cms.InputTag("voronoiBackgroundPF"),
                                      #srcRho = cms.InputTag("iterativeConePu5CaloJets","rho"),
                                      centrality = cms.InputTag("hiCentrality"),
                                    JetCorrections = cms.string(""),
                                    recoJetPtThreshold = cms.double(10),        
                                    genEnergyFractionThreshold = cms.double(0.05),
                                    genPtThreshold = cms.double(1.0),
                                    matchGenPtThreshold                 = cms.double(20.0),
                                    RThreshold = cms.double(0.3),
                                    reverseEnergyFractionThreshold = cms.double(0.5)
)
'''

JetAnalyzerAkVs3PF = cms.EDAnalyzer("JetTester_HeavyIons",
                                    JetType = cms.untracked.string('pf'),
                                    UEAlgo = cms.untracked.string('Vs'),
                                    OutputFile = cms.untracked.string(''),
                                    src = cms.InputTag("akVs3PFJets"),
                                    srcGen = cms.InputTag("ak3HiCleanedGenJets"),
                                    PFcands = cms.InputTag("particleFlowTmp"),
                                    Background = cms.InputTag("voronoiBackgroundPF"),
                                    #srcRho = cms.InputTag("akVs3PFJets","rho"),
                                    centrality = cms.InputTag("hiCentrality"),
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
                                    UEAlgo = cms.untracked.string('Vs'),
                                    OutputFile = cms.untracked.string(''),
                                    src = cms.InputTag("akVs4PFJets"),
                                    srcGen = cms.InputTag("ak4HiCleanedGenJets"),
                                    PFcands = cms.InputTag("particleFlowTmp"),
                                    Background = cms.InputTag("voronoiBackgroundPF"),
                                    #srcRho = cms.InputTag("iterativeConePu5CaloJets","rho"),
                                    centrality = cms.InputTag("hiCentrality"),
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
                                    UEAlgo = cms.untracked.string('Vs'),
                                    OutputFile = cms.untracked.string(''),
                                    src = cms.InputTag("akVs5PFJets"),
                                    srcGen = cms.InputTag("ak5HiCleanedGenJets"),
                                    PFcands = cms.InputTag("particleFlowTmp"),
                                    Background = cms.InputTag("voronoiBackgroundPF"),
                                      #srcRho = cms.InputTag("iterativeConePu5CaloJets","rho"),
                                    centrality = cms.InputTag("hiCentrality"),
                                    JetCorrections = cms.string(""),
                                    recoJetPtThreshold = cms.double(10),        
                                    genEnergyFractionThreshold = cms.double(0.05),
                                    genPtThreshold = cms.double(1.0),
                                    matchGenPtThreshold                 = cms.double(20.0),
                                    RThreshold = cms.double(0.3),
                                    reverseEnergyFractionThreshold = cms.double(0.5)
)
'''
JetAnalyzerAkVs6PF = cms.EDAnalyzer("JetTester_HeavyIons",
                                    JetType = cms.untracked.string('pf'),
                                    UEAlgo = cms.untracked.string('Vs'),
                                    OutputFile = cms.untracked.string(''),
                                    src = cms.InputTag("akVs6PFJets"),
                                    srcGen = cms.InputTag("ak6HiCleanedGenJets"),
                                      PFcands = cms.InputTag("particleFlowTmp"),
                                      Background = cms.InputTag("voronoiBackgroundPF"),
                                      #srcRho = cms.InputTag("iterativeConePu5CaloJets","rho"),
                                      centrality = cms.InputTag("hiCentrality"),
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
                                    UEAlgo = cms.untracked.string('Vs'),
                                    OutputFile = cms.untracked.string(''),
                                    src = cms.InputTag("akVs7PFJets"),
                                    srcGen = cms.InputTag("ak7HiCleanedGenJets"),
                                      PFcands = cms.InputTag("particleFlowTmp"),
                                      Background = cms.InputTag("voronoiBackgroundPF"),
                                      #srcRho = cms.InputTag("iterativeConePu5CaloJets","rho"),
                                      centrality = cms.InputTag("hiCentrality"),
                                    JetCorrections = cms.string(""),
                                    recoJetPtThreshold = cms.double(10),        
                                    genEnergyFractionThreshold = cms.double(0.05),
                                    genPtThreshold = cms.double(1.0),
                                    matchGenPtThreshold                 = cms.double(20.0),
                                    RThreshold = cms.double(0.3),
                                    reverseEnergyFractionThreshold = cms.double(0.5)
)				    			    

# AntiKt5 Calo jets
#-------------------------------------------------------------------------------
JetAnalyzerAk3Calo = cms.EDAnalyzer("JetTester",
                                    JetType = cms.untracked.string('calo'),
                                    OutputFile = cms.untracked.string(''),
                                    src            = cms.InputTag("ak3CaloJets"),
#                                    srcRho         = cms.InputTag("ro", "ak4CaloJets"),
                                    srcGen         = cms.InputTag("ak3HiCleanedGenJets"),
                                    JetCorrections = cms.string(""),
                                    recoJetPtThreshold = cms.double(40),
                                    genEnergyFractionThreshold     = cms.double(0.05),
                                    matchGenPtThreshold                 = cms.double(20.0),
                                    RThreshold                     = cms.double(0.3)
                                    )
#-------------------------------------------------------------------------------
JetAnalyzerAk4Calo = cms.EDAnalyzer("JetTester",
                                    JetType = cms.untracked.string('calo'),
                                    OutputFile = cms.untracked.string(''),
                                    src            = cms.InputTag("ak4CaloJets"),
#                                    srcRho         = cms.InputTag("rho", "ak4CaloJets"),
                                    srcGen         = cms.InputTag("ak4HiCleanedGenJets"),
                                    JetCorrections = cms.string(""),
                                    recoJetPtThreshold = cms.double(40),
                                    genEnergyFractionThreshold     = cms.double(0.05),
                                    matchGenPtThreshold                 = cms.double(20.0),
                                    RThreshold                     = cms.double(0.3)
                                    )
#-------------------------------------------------------------------------------
JetAnalyzerAk5Calo = cms.EDAnalyzer("JetTester",
                                    JetType = cms.untracked.string('calo'),
                                    OutputFile = cms.untracked.string(''),
                                    src            = cms.InputTag("ak5CaloJets"),
#                                    srcRho         = cms.InputTag("rho", "ak4CaloJets"),
                                    srcGen         = cms.InputTag("ak5HiCleanedGenJets"),
                                    JetCorrections = cms.string(""),
                                    recoJetPtThreshold = cms.double(40),
                                    genEnergyFractionThreshold     = cms.double(0.05),
                                    matchGenPtThreshold                 = cms.double(20.0),
                                    RThreshold                     = cms.double(0.3)
                                    )


# AntiKt5 PF jets
#-------------------------------------------------------------------------------
JetAnalyzerAk3PF = cms.EDAnalyzer("JetTester",
                                    JetType = cms.untracked.string('pf'),
                                    OutputFile = cms.untracked.string(''),
                                    src            = cms.InputTag("ak3PFJets"),
#                                    srcRho         = cms.InputTag("ro", "ak4PFJets"),
                                    srcGen         = cms.InputTag("ak3HiCleanedGenJets"),
                                    JetCorrections = cms.string(""),
                                    recoJetPtThreshold = cms.double(40),
                                    genEnergyFractionThreshold     = cms.double(0.05),
                                    matchGenPtThreshold                 = cms.double(20.0),
                                    RThreshold                     = cms.double(0.3)
                                    )
#-------------------------------------------------------------------------------
JetAnalyzerAk4PF = cms.EDAnalyzer("JetTester",
                                    JetType = cms.untracked.string('pf'),
                                    OutputFile = cms.untracked.string(''),
                                    src            = cms.InputTag("ak4PFJets"),
#                                    srcRho         = cms.InputTag("rho", "ak4PFJets"),
                                    srcGen         = cms.InputTag("ak4HiCleanedGenJets"),
                                    JetCorrections = cms.string(""),
                                    recoJetPtThreshold = cms.double(40),
                                    genEnergyFractionThreshold     = cms.double(0.05),
                                    matchGenPtThreshold                 = cms.double(20.0),
                                    RThreshold                     = cms.double(0.3)
                                    )
#-------------------------------------------------------------------------------
JetAnalyzerAk5PF = cms.EDAnalyzer("JetTester",
                                    JetType = cms.untracked.string('cpf'),
                                    OutputFile = cms.untracked.string(''),
                                    src            = cms.InputTag("ak5PFJets"),
#                                    srcRho         = cms.InputTag("rho", "ak4PFJets"),
                                    srcGen         = cms.InputTag("ak5HiCleanedGenJets"),
                                    JetCorrections = cms.string(""),
                                    recoJetPtThreshold = cms.double(40),
                                    genEnergyFractionThreshold     = cms.double(0.05),
                                    matchGenPtThreshold                 = cms.double(20.0),
                                    RThreshold                     = cms.double(0.3)
                                    )
'''
hiJetValidation = cms.Sequence(
    #ak2HiCleanedGenJets
    ak3HiCleanedGenJets
    #* ak4HiCleanedGenJets 
    #* ak5HiCleanedGenJets

    #* iterativeCone7HiCleanedGenJets
    #* iterativeCone5HiCleanedGenJets
    #* JetAnalyzerICPU7Calo
    #* JetAnalyzerICPU5Calo

    * JetAnalyzerAkPU3Calo
    #* JetAnalyzerAkPU4Calo
    #* JetAnalyzerAkPU5Calo

    * JetAnalyzerAkPU3PF
    #* JetAnalyzerAkPU4PF
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

    #* JetAnalyzerAk3Calo
    #* JetAnalyzerAk4Calo
    #* JetAnalyzerAk5Calo

    #* JetAnalyzerAk3PF
    #* JetAnalyzerAk4PF
    #* JetAnalyzerAk5PF    

)
