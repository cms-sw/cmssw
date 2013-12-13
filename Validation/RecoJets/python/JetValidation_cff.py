import FWCore.ParameterSet.Config as cms

# File: JetValidation_cff.py
# Author : Chiyoung Jeong
# Date : Mar 17 2009
# Description : cff file for DQM offline vladation


from RecoJets.Configuration.RecoJetAssociations_cff import *

from Validation.RecoJets.JetValidation_cfi import *

from JetMETCorrections.Configuration.JetCorrectionServices_cff import ak4CaloL2L3,ak4CaloL2Relative,ak4CaloL3Absolute
newAk5CaloL2L3 = ak4CaloL2L3.clone()

#from JetMETCorrections.Configuration.JetCorrectionServicesAllAlgos_cff import ak8CaloL2L3,ak8CaloL2Relative,ak8CaloL3Absolute
#newAk7CaloL2L3 = ak8CaloL2L3.clone()

from JetMETCorrections.Configuration.JetCorrectionServices_cff import ak4PFL1FastL2L3,ak4PFL1Fastjet,ak4PFL2Relative,ak4PFL3Absolute
newAk5PFL1FastL2L3 = ak4PFL1FastL2L3.clone()

from JetMETCorrections.Configuration.JetCorrectionServices_cff import ak4JPTL1FastL2L3,ak4JPTL1Fastjet,ak4JPTL2Relative,ak4JPTL3Absolute
newAk5JPTL1FastL2L3 = ak4JPTL1FastL2L3.clone()

from JetCorrectionServices_AK4CHS_cff import ak4PFchsL1FastL2L3,ak4PFchsL1Fastjet,ak4PFchsL2Relative,ak4PFchsL3Absolute
newAk5PFchsL1FastL2L3 = ak4PFchsL1FastL2L3.clone()

JetValidation = cms.Sequence(
#    JetAnalyzerKt6PF*
#    JetAnalyzerKt6Calo*
    JetAnalyzerAk5Calo*
#    JetAnalyzerAk7Calo*
    JetAnalyzerAk5PF*
    JetAnalyzerAk5JPT*
    JetAnalyzerAk5PFCHS
#    JetAnalyzerAk8PF*
#    JetAnalyzerAk8PFCHS*
#    JetAnalyzerCA8PFCHS
    
    )
