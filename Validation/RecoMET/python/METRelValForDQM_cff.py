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

from JetMETCorrections.Configuration.JetCorrectors_cff import ak4PFL1FastL2L3CorrectorChain,ak4PFL1FastL2L3Corrector,ak4PFL3AbsoluteCorrector,ak4PFL2RelativeCorrector,ak4PFL1FastjetCorrector

newAK4PFL1FastL2L3Corrector = ak4PFL1FastL2L3Corrector.clone()
newAK4PFL1FastL2L3CorrectorChain = cms.Sequence(
    #ak4PFL1FastjetCorrector * ak4PFL2RelativeCorrector * ak4PFL3AbsoluteCorrector * 
    newAK4PFL1FastL2L3Corrector
    )

metPreValidSeq=cms.Sequence(ak4PFL1FastjetCorrector * ak4PFL2RelativeCorrector * ak4PFL3AbsoluteCorrector)

valCorrPfMetType1=corrPfMetType1.clone(jetCorrLabel = cms.InputTag('newAK4PFL1FastL2L3Corrector'))

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
    #metHOAnalyzer*
    #metNoHFAnalyzer*
    #metNoHFHOAnalyzer*
    #metOptAnalyzer*
    #metOptHOAnalyzer*
    #metOptNoHFAnalyzer*
    #metOptNoHFHOAnalyzer
    pfMetAnalyzer*
    #tcMetAnalyzer*
    #corMetGlobalMuonsAnalyzer*
    genMetTrueAnalyzer*
    #genMetCaloAnalyzer*
    #genMetCaloAndNonPromptAnalyzer
    correctionTermsPfMetType0PFCandidateForValidation*
    newAK4PFL1FastL2L3CorrectorChain*
    valCorrPfMetType1*
    #pfchsMETcorr*
    pfMetT0pc*
    PfMetT1*
    PfMetT0pcT1*
    pfType0CorrectedMetAnalyzer*
    pfType1CorrectedMetAnalyzer*
    pfType01CorrectedMetAnalyzer
	 )


METValidation = cms.Sequence(
    metAnalyzer*
    #metHOAnalyzer*
    #metNoHFAnalyzer*
    #metNoHFHOAnalyzer*
    #metOptAnalyzer*
    #metOptHOAnalyzer*
    #metOptNoHFAnalyzer*
    #metOptNoHFHOAnalyzer*
    pfMetAnalyzer*
    #tcMetAnalyzer*
    #corMetGlobalMuonsAnalyzer*
    genMetTrueAnalyzer*#*
    #genMetCaloAnalyzer*
    #genMetCaloAndNonPromptAnalyzer
    correctionTermsPfMetType0PFCandidateForValidation*
    newAK4PFL1FastL2L3CorrectorChain*
    valCorrPfMetType1*
    #pfchsMETcorr*
    pfMetT0pc*
    PfMetT1*
    PfMetT0pcT1*
    pfType0CorrectedMetAnalyzer*
    pfType1CorrectedMetAnalyzer*
    pfType01CorrectedMetAnalyzer
    )

METValidationMiniAOD = cms.Sequence(pfType1CorrectedMetAnalyzerMiniAOD)
