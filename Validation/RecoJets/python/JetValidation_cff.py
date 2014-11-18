import FWCore.ParameterSet.Config as cms

# File: JetValidation_cff.py
# Author : Chiyoung Jeong
# Date : Mar 17 2009
# Description : cff file for DQM offline vladation


from RecoJets.Configuration.RecoJetAssociations_cff import *

from Validation.RecoJets.JetValidation_cfi import *

from JetMETCorrections.Configuration.JetCorrectors_cff import ak4CaloL2L3CorrectorChain,ak4CaloL2L3Corrector,ak4CaloL3AbsoluteCorrector,ak4CaloL2RelativeCorrector

from JetMETCorrections.Configuration.JetCorrectors_cff import ak4PFL1FastL2L3CorrectorChain,ak4PFL1FastL2L3Corrector,ak4PFL3AbsoluteCorrector,ak4PFL2RelativeCorrector,ak4PFL1FastjetCorrector

from JetMETCorrections.Configuration.JetCorrectors_cff import ak4PFCHSL1FastL2L3CorrectorChain,ak4PFCHSL1FastL2L3Corrector,ak4PFCHSL3AbsoluteCorrector,ak4PFCHSL2RelativeCorrector,ak4PFCHSL1FastjetCorrector

#from JetMETCorrections.Configuration.JetCorrectionServices_cff import ak4CaloL2L3,ak4CaloL2Relative,ak4CaloL3Absolute
#newAk4CaloL2L3 = ak4CaloL2L3.clone()

#from JetMETCorrections.Configuration.JetCorrectionServicesAllAlgos_cff import ak7CaloL2L3,ak7CaloL2Relative,ak7CaloL3Absolute
#newAk7CaloL2L3 = ak7CaloL2L3.clone()

#from JetMETCorrections.Configuration.JetCorrectionServices_cff import ak4PFL1FastL2L3,ak4PFL1Fastjet,ak4PFL2Relative,ak4PFL3Absolute
#newAk4PFL1FastL2L3 = ak4PFL1FastL2L3.clone()

#from JetMETCorrections.Configuration.JetCorrectionServices_cff import ak4JPTL1FastL2L3,ak4JPTL1Fastjet,ak4JPTL2Relative,ak4JPTL3Absolute
#newAk4JPTL1FastL2L3 = ak4JPTL1FastL2L3.clone()

#from JetMETCorrections.Configuration.JetCorrectionServices_cff import ak4PFCHSL1FastL2L3,ak4PFCHSL1Fastjet,ak4PFCHSL2Relative,ak4PFCHSL3Absolute

JetValidation = cms.Sequence(
#    JetAnalyzerKt6PF*
#    JetAnalyzerKt6Calo*
    ak4CaloL2L3CorrectorChain*
    JetAnalyzerAk4Calo*
#    JetAnalyzerAk7Calo*
    ak4PFL1FastL2L3CorrectorChain*
    JetAnalyzerAk4PF*
#    JetAnalyzerAk4JPT*
    ak4PFCHSL1FastL2L3CorrectorChain*
    JetAnalyzerAk4PFCHS
#    JetAnalyzerAk8PF*
#    JetAnalyzerAk8PFCHS*
#    *JetAnalyzerCA8PFCHS
    )

JetValidationMiniAOD=cms.Sequence(JetAnalyzerAk4PFCHSMiniAOD)
