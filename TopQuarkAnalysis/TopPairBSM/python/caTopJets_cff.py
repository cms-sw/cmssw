import FWCore.ParameterSet.Config as cms

from TopQuarkAnalysis.TopPairBSM.CATopJetParameters_cfi import *
from RecoJets.JetProducers.GenJetParameters_cfi import *
from RecoJets.JetProducers.CaloJetParameters_cfi import *


caTopJetsProducer = cms.EDProducer("CATopJetProducer",
                                   CATopJetParameters,
                                   CaloJetParameters
                                   )

