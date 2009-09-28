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

JetValidation = cms.Sequence(
                       process.JetAnalyzerIC5Calo
                      *process.JetAnalyzerIC5PF
                      *process.JetAnalyzerKt4Calo*process.JetAnalyzerKt6Calo
                      *process.JetAnalyzerSc5Calo*process.JetAnalyzerSc7Calo
                      *process.JetAnalyzerAk5Calo*process.JetAnalyzerAk7Calo
                      *process.JetAnalyzerAk5PF)
