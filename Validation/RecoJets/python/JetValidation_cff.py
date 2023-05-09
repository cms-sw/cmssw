import FWCore.ParameterSet.Config as cms

# File: JetValidation_cff.py
# Author : Chiyoung Jeong
# Date : Mar 17 2009
# Description : cff file for DQM offline vladation


from RecoJets.Configuration.RecoJetAssociations_cff import *

from Validation.RecoJets.JetValidation_cfi import *

from JetMETCorrections.Configuration.JetCorrectors_cff import ak4CaloL2L3CorrectorChain,ak4CaloL2L3Corrector,ak4CaloL3AbsoluteCorrector,ak4CaloL2RelativeCorrector

newAk4CaloL2L3Corrector = ak4CaloL2L3Corrector.clone()
newAk4CaloL2L3CorrectorChain = cms.Sequence(
    #ak4CaloL2RelativeCorrector * ak4CaloL3AbsoluteCorrector * 
    newAk4CaloL2L3Corrector
)

from JetMETCorrections.Configuration.JetCorrectors_cff import ak4PFL1FastL2L3CorrectorChain,ak4PFL1FastL2L3Corrector,ak4PFL3AbsoluteCorrector,ak4PFL2RelativeCorrector,ak4PFL1FastjetCorrector

newAk4PFL1FastL2L3Corrector = ak4PFL1FastL2L3Corrector.clone()
newAk4PFL1FastL2L3CorrectorChain = cms.Sequence(
    #ak4PFL1FastjetCorrector * ak4PFL2RelativeCorrector * ak4PFL3AbsoluteCorrector * 
    newAk4PFL1FastL2L3Corrector
)

from JetMETCorrections.Configuration.JetCorrectors_cff import ak4PFCHSL1FastL2L3CorrectorChain,ak4PFCHSL1FastL2L3Corrector,ak4PFCHSL3AbsoluteCorrector,ak4PFCHSL2RelativeCorrector,ak4PFCHSL1FastjetCorrector

newAk4PFCHSL1FastL2L3Corrector = ak4PFCHSL1FastL2L3Corrector.clone()
newAk4PFCHSL1FastL2L3CorrectorChain = cms.Sequence(
    #ak4PFCHSL1FastjetCorrector * ak4PFCHSL2RelativeCorrector * ak4PFCHSL3AbsoluteCorrector * 
    newAk4PFCHSL1FastL2L3Corrector
)

jetPreValidSeqTask = cms.Task(ak4CaloL2RelativeCorrector,
                              ak4CaloL3AbsoluteCorrector,
                              ak4PFL1FastjetCorrector,
                              ak4PFL2RelativeCorrector,
                              ak4PFL3AbsoluteCorrector,
                              ak4PFCHSL1FastjetCorrector,
                              ak4PFCHSL2RelativeCorrector,
                              ak4PFCHSL3AbsoluteCorrector
)
jetPreValidSeq=cms.Sequence(jetPreValidSeqTask)

JetValidation = cms.Sequence(
#    JetAnalyzerKt6PF*
#    JetAnalyzerKt6Calo*
    newAk4CaloL2L3CorrectorChain*
    JetAnalyzerAk4Calo*
#    JetAnalyzerAk7Calo*
    newAk4PFL1FastL2L3CorrectorChain*
    JetAnalyzerAk4PF*
#    JetAnalyzerAk4JPT*
    newAk4PFCHSL1FastL2L3CorrectorChain*
    JetAnalyzerAk4PFCHS
#    JetAnalyzerAk8PF*
#    JetAnalyzerAk8PFCHS*
#    *JetAnalyzerCA8PFCHS
    )


JetValidationMiniAOD=cms.Sequence(JetAnalyzerAk4PFCHSMiniAOD*JetAnalyzerAk4PFPUPPIMiniAOD*JetAnalyzerAk8PFPUPPIMiniAOD)
