import FWCore.ParameterSet.Config as cms
# File: JetValidation_cff.py
# Author : Chiyoung Jeong
# Date : Mar 17 2009
# Description : cff file for DQM offline vladation.


from RecoJets.Configuration.RecoJetAssociations_cff import *

from Validation.RecoJets.JetValidation_cfi import *

from JetMETCorrections.Configuration.JetCorrectionServices_cff import ak5CaloL2L3,ak5CaloL2Relative,ak5CaloL3Absolute
newAk5CaloL2L3 = ak5CaloL2L3.clone()

from JetMETCorrections.Configuration.JetCorrectionServicesAllAlgos_cff import ak7CaloL2L3,ak7CaloL2Relative,ak7CaloL3Absolute
newAk7CaloL2L3 = ak7CaloL2L3.clone()

from JetMETCorrections.Configuration.JetCorrectionServices_cff import ak5PFL1FastL2L3,ak5PFL1Fastjet,ak5PFL2Relative,ak5PFL3Absolute
newAk5PFL1FastL2L3 = ak5PFL1FastL2L3.clone()

from JetMETCorrections.Configuration.JetCorrectionServices_cff import ak5JPTL1FastL2L3,ak5JPTL1Fastjet,ak5JPTL2Relative,ak5JPTL3Absolute
newAk5JPTL1FastL2L3 = ak5JPTL1FastL2L3.clone()

JetValidation = cms.Sequence(
#                      JetAnalyzerIC5Calo*
#                      JetAnalyzerIC5PF*
                      JetAnalyzerKt6PF*JetAnalyzerKt6Calo*
                      JetAnalyzerAk5Calo*JetAnalyzerAk7Calo
                      *JetAnalyzerAk5PF
                      *JetAnalyzerAk5JPT
#                      *JetAnalyzerIC5JPT
)
