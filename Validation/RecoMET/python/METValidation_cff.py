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

from JetMETCorrections.Type1MET.pfMETCorrections_cff import pfJetMETcorr, pfchsMETcorr, pfType1CorrectedMet 

from JetMETCorrections.Configuration.JetCorrectionServices_cff import ak5PFL1FastL2L3,ak5PFL1Fastjet,ak5PFL2Relative,ak5PFL3Absolute
newAk5PFL1FastL2L3 = ak5PFL1FastL2L3.clone()
pfJetMETcorr.jetCorrLabel = cms.string('newAk5PFL1FastL2L3')

pfType0CorrectedMet = pfType1CorrectedMet.clone(applyType0Corrections = cms.bool(True), applyType1Corrections = cms.bool(False))
pfType01CorrectedMet = pfType1CorrectedMet.clone(applyType0Corrections = cms.bool(True), applyType1Corrections = cms.bool(True))

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
    tcMetAnalyzer*
    #corMetGlobalMuonsAnalyzer*
    genMetTrueAnalyzer*
    #genMetCaloAnalyzer*
    #genMetCaloAndNonPromptAnalyzer
    pfJetMETcorr*
    pfchsMETcorr*
    pfType0CorrectedMet*
    pfType1CorrectedMet*
    pfType01CorrectedMet*
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
    tcMetAnalyzer*
    #corMetGlobalMuonsAnalyzer*
    genMetTrueAnalyzer*#*
    #genMetCaloAnalyzer*
    #genMetCaloAndNonPromptAnalyzer
    pfJetMETcorr*
    pfchsMETcorr*
    pfType0CorrectedMet*
    pfType1CorrectedMet*
    pfType01CorrectedMet*
    pfType0CorrectedMetAnalyzer*
    pfType1CorrectedMetAnalyzer*
    pfType01CorrectedMetAnalyzer
    )

    


