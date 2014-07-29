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


from JetMETCorrections.Configuration.JetCorrectionServices_cff import ak4PFL1FastL2L3,ak4PFL1Fastjet,ak4PFL2Relative,ak4PFL3Absolute
newAK4PFL1FastL2L3 = ak4PFL1FastL2L3.clone()
corrPfMetType1.jetCorrLabel = cms.string('newAK4PFL1FastL2L3')

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
    correctionTermsPfMetType0PFCandidate*
    corrPfMetType1*
    #pfchsMETcorr*
    pfMetT0pc*
    pfMetT1*
    pfMetT0pcT1*
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
    correctionTermsPfMetType0PFCandidate*
    corrPfMetType1*
    #pfchsMETcorr*
    pfMetT0pc*
    pfMetT1*
    pfMetT0pcT1*
    pfType0CorrectedMetAnalyzer*
    pfType1CorrectedMetAnalyzer*
    pfType01CorrectedMetAnalyzer
    )




