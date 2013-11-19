import FWCore.ParameterSet.Config as cms
# File: METRelValForDQM.cff
# Author:  R. Remington
# Date: 03.01.09
# Fill validation histograms for MET.

from Validation.RecoMET.METValidation_cfi import *
#metAnalyzer.FolderName = cms.untracked.string("JetMET/METv/")

#metHOAnalyzer.FolderName = cms.untracked.string("JetMET/METv/")

#metNoHFAnalyzer.FolderName = cms.untracked.string("JetMET/METv/")

#metNoHFHOAnalyzer.FolderName = cms.untracked.string("JetMET/METv/")

#metOptAnalyzer.FolderName = cms.untracked.string("JetMET/METv/")

#metOptHOAnalyzer.FolderName = cms.untracked.string("JetMET/METv/")

#metOptNoHFAnalyzer.FolderName = cms.untracked.string("JetMET/METv/")

#metOptNoHFHOAnalyzer.FolderName = cms.untracked.string("JetMET/METv/")

#from Validation.RecoMET.PFMET_cfi import *
#pfMetAnalyzer.FolderName = cms.untracked.string("JetMET/METv/")

#from Validation.RecoMET.TCMET_cfi import *
#tcMetAnalyzer.FolderName = cms.untracked.string("JetMET/METv/")

#from Validation.RecoMET.MuonCorrectedCaloMET_cff import *
#corMetGlobalMuonsAnalyzer.FolderName = cms.untracked.string("JetMET/METv/")

#from Validation.RecoMET.GenMET_cfi import *
#genMetTrueAnalyzer.FolderName = cms.untracked.string("JetMET/METv/")

#genMetCaloAnalyzer.FolderName = cms.untracked.string("JetMET/METv/")

#genMetCaloAndNonPromptAnalyzer.FolderName = cms.untracked.string("JetMET/METv/")


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

    


