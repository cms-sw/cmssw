import FWCore.ParameterSet.Config as cms
# File: JetValidation_cff.py
# Author : Chiyoung Jeong
# Date : Mar 17 2009
# Description : cff file for DQM offline vladation.


from JetMETCorrections.Configuration.JetCorrectionsRecord_cfi import *
from RecoJets.Configuration.RecoJetAssociations_cff import *

from JetMETCorrections.Configuration.JetPlusTrackCorrections_cff import *
from JetMETCorrections.Configuration.ZSPJetCorrections219_cff import *
from JetMETCorrections.Configuration.L2L3Corrections_Summer08Redigi_cff import *

from Validation.RecoJets.JetValidation_cfi import *

#prefer("L2L3JetCorrectorIcone5")

L2L3CorJetIcone5 = cms.EDProducer("CaloJetCorrectionProducer",
    src = cms.InputTag("iterativeCone5CaloJets"),
    correctors = cms.vstring('L2L3JetCorrectorIcone5')
)


JetValidation = cms.Sequence(L2L3CorJetIcone5*ZSPJetCorrections*JetPlusTrackCorrections*JetAnalyzer1*JetAnalyzer2*JetAnalyzer3*JetAnalyzer4*JetAnalyzer5*JetAnalyzer6*JetAnalyzer7*JetAnalyzer8)
