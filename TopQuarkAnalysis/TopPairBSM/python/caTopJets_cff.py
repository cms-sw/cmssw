import FWCore.ParameterSet.Config as cms

#from RecoJets.JetProducers.CaloJetParameters_cfi import *
from TopQuarkAnalysis.TopPairBSM.CATopJetParameters_cfi import *


caTopJetsProducer = cms.EDProducer("CATopJetProducer",
                           CATopJetParameters
                           )

