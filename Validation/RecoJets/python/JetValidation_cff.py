import FWCore.ParameterSet.Config as cms
# File: JetValidation_cff.py
# Author : Chiyoung Jeong
# Date : Mar 17 2009
# Description : cff file for DQM offline vladation.


from JetMETCorrections.Configuration.JetCorrectionsRecord_cfi import *
from RecoJets.Configuration.RecoJetAssociations_cff import *

from Validation.RecoJets.JetValidation_cfi import *

JetValidation = cms.Sequence(
                       JetAnalyzerIC5Calo
                      *JetAnalyzerIC5PF
                      *JetAnalyzerKt4Calo*JetAnalyzerKt6Calo
                      *JetAnalyzerSc5Calo*JetAnalyzerSc7Calo
                      *JetAnalyzerAk5Calo*JetAnalyzerAk7Calo
                      *JetAnalyzerAk5PF)
