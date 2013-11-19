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
    genMetTrueAnalyzer#*
    #genMetCaloAnalyzer*
    #genMetCaloAndNonPromptAnalyzer
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
    genMetTrueAnalyzer#*
    #genMetCaloAnalyzer*
    #genMetCaloAndNonPromptAnalyzer
    )

    


