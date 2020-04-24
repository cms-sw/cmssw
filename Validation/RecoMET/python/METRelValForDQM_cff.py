import FWCore.ParameterSet.Config as cms
# File: METRelValForDQM.cff
# Author:  R. Remington
# Date: 03.01.09
# Fill validation histograms for MET.
from Validation.RecoMET.METValidation_cfi import *

#Removed the MET collections that we no longer monitor
#in an attempt to reduce the number of histograms produced
# as requested by DQM group to reduce the load on server.
# -Samantha Hewamanage (samantha@cern.ch) - 04-27-2012

from JetMETCorrections.Type1MET.correctedMet_cff import pfMetT0pc,pfMetT0pcT1,pfMetT1
from JetMETCorrections.Type1MET.correctionTermsPfMetType0PFCandidate_cff import *
from JetMETCorrections.Type1MET.correctionTermsPfMetType1Type2_cff import corrPfMetType1

from JetMETCorrections.Configuration.JetCorrectors_cff import ak4PFCHSL1FastL2L3ResidualCorrectorChain,ak4PFCHSL1FastL2L3CorrectorChain,ak4PFCHSL1FastL2L3ResidualCorrector,ak4PFCHSResidualCorrector,ak4PFCHSL1FastL2L3Corrector,ak4PFCHSL3AbsoluteCorrector,ak4PFCHSL2RelativeCorrector,ak4PFCHSL1FastjetCorrector

newAK4PFCHSL1FastL2L3Corrector = ak4PFCHSL1FastL2L3Corrector.clone()
newAK4PFCHSL1FastL2L3CorrectorChain = cms.Sequence(
    #ak4PFCHSL1FastjetCorrector * ak4PFCHSL2RelativeCorrector * ak4PFCHSL3AbsoluteCorrector * 
    newAK4PFCHSL1FastL2L3Corrector
    )

newAK4PFCHSL1FastL2L3ResidualCorrector = ak4PFCHSL1FastL2L3ResidualCorrector.clone()
newAK4PFCHSL1FastL2L3ResidualCorrectorChain = cms.Sequence(
    #ak4PFCHSL1FastjetCorrector * ak4PFCHSL2RelativeCorrector * ak4PFCHSL3AbsoluteCorrector * 
    newAK4PFCHSL1FastL2L3ResidualCorrector
    )

metPreValidSeqTask = cms.Task(ak4PFCHSL1FastjetCorrector,
                              ak4PFCHSL2RelativeCorrector,
                              ak4PFCHSL3AbsoluteCorrector,
                              ak4PFCHSResidualCorrector
)
metPreValidSeq = cms.Sequence(metPreValidSeqTask)

valCorrPfMetType1=corrPfMetType1.clone(jetCorrLabel = cms.InputTag('newAK4PFCHSL1FastL2L3Corrector'),
                                       jetCorrLabelRes = cms.InputTag('newAK4PFCHSL1FastL2L3ResidualCorrector')
                                      )

PfMetT1=pfMetT1.clone(srcCorrections = cms.VInputTag(
        cms.InputTag('valCorrPfMetType1', 'type1')
        ))

PfMetT0pcT1=pfMetT0pcT1.clone(
    srcCorrections = cms.VInputTag(
        cms.InputTag('corrPfMetType0PfCand'),
        cms.InputTag('valCorrPfMetType1', 'type1')
        )
    )

METRelValSequence = cms.Sequence(
    metAnalyzer*
    pfMetAnalyzer*
    genMetTrueAnalyzer*
    correctionTermsPfMetType0PFCandidateForValidation*
    newAK4PFCHSL1FastL2L3CorrectorChain*
    newAK4PFCHSL1FastL2L3ResidualCorrectorChain*
    valCorrPfMetType1*
    pfMetT0pc*
    PfMetT1*
    PfMetT0pcT1*
    pfType0CorrectedMetAnalyzer*
    pfType1CorrectedMetAnalyzer*
    pfType01CorrectedMetAnalyzer
	 )


METValidation = cms.Sequence(
    metAnalyzer*
    pfMetAnalyzer*
    genMetTrueAnalyzer*
    correctionTermsPfMetType0PFCandidateForValidation*
    newAK4PFCHSL1FastL2L3CorrectorChain*
    newAK4PFCHSL1FastL2L3ResidualCorrectorChain*
    valCorrPfMetType1*
    pfMetT0pc*
    PfMetT1*
    PfMetT0pcT1*
    pfType0CorrectedMetAnalyzer*
    pfType1CorrectedMetAnalyzer*
    pfType01CorrectedMetAnalyzer
    )

METValidationMiniAOD = cms.Sequence(pfType1CorrectedMetAnalyzerMiniAOD*pfPuppiMetAnalyzerMiniAOD)
