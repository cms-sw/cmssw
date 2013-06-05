import FWCore.ParameterSet.Config as cms

process = cms.Process("TrackingTruthOutputTest")
process.load("FWCore.MessageService.MessageLogger_cfi")

# Standard includes
process.load('Configuration/StandardSequences/GeometryIdeal_cff')
process.load("Configuration.StandardSequences.RawToDigi_cff")
process.load("Configuration.EventContent.EventContent_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("SimGeneral.MixingModule.mixNoPU_cfi")

# Conditions
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'STARTUP31X_V2::All'

# Playback
process.load("SimGeneral.TrackingAnalysis.Playback_cfi")

# Validation-specific includes

process.load("SimTracker.TrackAssociation.TrackAssociatorByHits_cfi")
process.load("Validation.RecoTrack.cuts_cff")
process.load("Validation.RecoTrack.MultiTrackValidator_cff")
process.load("SimGeneral.TrackingAnalysis.trackingParticles_cfi")
process.load("DQMServices.Components.EDMtoMEConverter_cff")
process.load("Validation.Configuration.postValidation_cff")

# TrackHistory includes

process.load("SimTracker.TrackHistory.TrackClassifier_cff")

from SimTracker.TrackHistory.CategorySelectors_cff import *

# Configuration MultiTrackValidator

process.multiTrackValidator.outputFile = 'TrackingTruthValidation.root'
process.multiTrackValidator.associators = ['TrackAssociatorByHits']
process.multiTrackValidator.skipHistoFit = cms.untracked.bool(False)
process.multiTrackValidator.useLogPt = cms.untracked.bool(True)
process.multiTrackValidator.minpT = cms.double(-1)
process.multiTrackValidator.maxpT = cms.double(3)
process.multiTrackValidator.nintpT = cms.int32(40)

process.cutsRecoTracks.quality = cms.vstring('highPurity')
process.multiTrackValidator.label = ['cutsRecoTracks']
process.multiTrackValidator.UseAssociators = cms.bool(True)

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)

# Configuration for TrackHistory 

process.add_( 
  cms.Service("TFileService",
      fileName = cms.string("TrackingTruthHistory.root")
  )
)

process.trackCategorySelector = TrackingParticleCategorySelector(
    src = cms.InputTag('mix', 'MergedTrackTruth'),
    cut = cms.string("is('SignalEvent')")
)

process.trackingParticleCategoriesAnalyzer = cms.EDAnalyzer("TrackingParticleCategoriesAnalyzer",
    process.trackClassifier
)

process.trackingParticleCategoriesAnalyzer.trackingTruth = 'trackCategorySelector'

# Sequences and paths

process.trackingTruth = cms.Sequence(
    process.mix * process.trackingParticles
)

process.validation = cms.Sequence(
    process.cutsRecoTracks * process.multiTrackValidator * process.trackCategorySelector * process.trackingParticleCategoriesAnalyzer
)

process.p = cms.Path(process.trackingTruth * process.validation)

# Input definition
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(5000) )
readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring() 
process.source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)

readFiles.extend( [
       '/store/relval/CMSSW_3_2_2/RelValTTbar_Tauola/GEN-SIM-RECO/STARTUP31X_V2_156BxLumiPileUp-v1/0008/8ADB0D58-967C-DE11-B9E8-001D09F2305C.root',
       '/store/relval/CMSSW_3_2_2/RelValTTbar_Tauola/GEN-SIM-RECO/STARTUP31X_V2_156BxLumiPileUp-v1/0008/2093B78A-DA7C-DE11-A400-001D09F2437B.root',
       '/store/relval/CMSSW_3_2_2/RelValTTbar_Tauola/GEN-SIM-RECO/STARTUP31X_V2_156BxLumiPileUp-v1/0007/F6E7AC61-4D7C-DE11-A86F-001D09F2A465.root',
       '/store/relval/CMSSW_3_2_2/RelValTTbar_Tauola/GEN-SIM-RECO/STARTUP31X_V2_156BxLumiPileUp-v1/0007/E6938DB2-4D7C-DE11-A56C-000423D991D4.root',
       '/store/relval/CMSSW_3_2_2/RelValTTbar_Tauola/GEN-SIM-RECO/STARTUP31X_V2_156BxLumiPileUp-v1/0007/BAD7002F-507C-DE11-BE09-001D09F29597.root',
       '/store/relval/CMSSW_3_2_2/RelValTTbar_Tauola/GEN-SIM-RECO/STARTUP31X_V2_156BxLumiPileUp-v1/0007/ACCA8D9C-4D7C-DE11-83F8-001D09F24D67.root',
       '/store/relval/CMSSW_3_2_2/RelValTTbar_Tauola/GEN-SIM-RECO/STARTUP31X_V2_156BxLumiPileUp-v1/0007/96A9C809-5A7C-DE11-A3C2-000423D98950.root',
       '/store/relval/CMSSW_3_2_2/RelValTTbar_Tauola/GEN-SIM-RECO/STARTUP31X_V2_156BxLumiPileUp-v1/0007/847835A2-4C7C-DE11-8EB7-000423D9880C.root',
       '/store/relval/CMSSW_3_2_2/RelValTTbar_Tauola/GEN-SIM-RECO/STARTUP31X_V2_156BxLumiPileUp-v1/0007/54493560-527C-DE11-B497-001D09F25109.root',
       '/store/relval/CMSSW_3_2_2/RelValTTbar_Tauola/GEN-SIM-RECO/STARTUP31X_V2_156BxLumiPileUp-v1/0007/524126BE-4E7C-DE11-B020-000423D9880C.root',
       '/store/relval/CMSSW_3_2_2/RelValTTbar_Tauola/GEN-SIM-RECO/STARTUP31X_V2_156BxLumiPileUp-v1/0007/16C79B9F-6C7C-DE11-9428-000423D174FE.root'
] )

secFiles.extend( [
       '/store/relval/CMSSW_3_2_2/RelValTTbar_Tauola/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V2_156BxLumiPileUp-v1/0008/9A0ED789-DA7C-DE11-B6D3-001D09F29619.root',
       '/store/relval/CMSSW_3_2_2/RelValTTbar_Tauola/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V2_156BxLumiPileUp-v1/0008/38598951-967C-DE11-8578-001D09F24D8A.root',
       '/store/relval/CMSSW_3_2_2/RelValTTbar_Tauola/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V2_156BxLumiPileUp-v1/0007/FCAA7A27-557C-DE11-8064-000423D99658.root',
       '/store/relval/CMSSW_3_2_2/RelValTTbar_Tauola/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V2_156BxLumiPileUp-v1/0007/F09E1E99-4D7C-DE11-B7F0-001D09F29619.root',
       '/store/relval/CMSSW_3_2_2/RelValTTbar_Tauola/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V2_156BxLumiPileUp-v1/0007/D6E18955-527C-DE11-A90C-001D09F24691.root',
       '/store/relval/CMSSW_3_2_2/RelValTTbar_Tauola/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V2_156BxLumiPileUp-v1/0007/C851451B-4C7C-DE11-916F-000423D944F0.root',
       '/store/relval/CMSSW_3_2_2/RelValTTbar_Tauola/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V2_156BxLumiPileUp-v1/0007/BADEB400-5F7C-DE11-8CFB-000423D98EA8.root',
       '/store/relval/CMSSW_3_2_2/RelValTTbar_Tauola/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V2_156BxLumiPileUp-v1/0007/B4E20C35-4D7C-DE11-96C6-000423D95030.root',
       '/store/relval/CMSSW_3_2_2/RelValTTbar_Tauola/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V2_156BxLumiPileUp-v1/0007/AC51B347-507C-DE11-BA3E-001D09F29533.root',
       '/store/relval/CMSSW_3_2_2/RelValTTbar_Tauola/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V2_156BxLumiPileUp-v1/0007/8E540997-4D7C-DE11-9DE7-000423D6CA42.root',
       '/store/relval/CMSSW_3_2_2/RelValTTbar_Tauola/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V2_156BxLumiPileUp-v1/0007/8CAFCA68-4D7C-DE11-B74D-000423D987E0.root',
       '/store/relval/CMSSW_3_2_2/RelValTTbar_Tauola/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V2_156BxLumiPileUp-v1/0007/8C80A0DF-4D7C-DE11-B927-000423D98BC4.root',
       '/store/relval/CMSSW_3_2_2/RelValTTbar_Tauola/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V2_156BxLumiPileUp-v1/0007/82F85C38-5C7C-DE11-9560-000423D6B444.root',
       '/store/relval/CMSSW_3_2_2/RelValTTbar_Tauola/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V2_156BxLumiPileUp-v1/0007/800E8389-717C-DE11-BB29-001D09F251FE.root',
       '/store/relval/CMSSW_3_2_2/RelValTTbar_Tauola/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V2_156BxLumiPileUp-v1/0007/748AEF76-4D7C-DE11-8476-000423D60FF6.root',
       '/store/relval/CMSSW_3_2_2/RelValTTbar_Tauola/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V2_156BxLumiPileUp-v1/0007/6A74F5A3-6C7C-DE11-8818-001D09F29597.root',
       '/store/relval/CMSSW_3_2_2/RelValTTbar_Tauola/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V2_156BxLumiPileUp-v1/0007/6240240D-4D7C-DE11-A4B2-001D09F29619.root',
       '/store/relval/CMSSW_3_2_2/RelValTTbar_Tauola/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V2_156BxLumiPileUp-v1/0007/620E14B8-4D7C-DE11-84D7-000423D9517C.root',
       '/store/relval/CMSSW_3_2_2/RelValTTbar_Tauola/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V2_156BxLumiPileUp-v1/0007/60CFC2A5-4C7C-DE11-B10F-001D09F2A465.root',
       '/store/relval/CMSSW_3_2_2/RelValTTbar_Tauola/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V2_156BxLumiPileUp-v1/0007/5E3D53A2-4E7C-DE11-93B5-001D09F29533.root',
       '/store/relval/CMSSW_3_2_2/RelValTTbar_Tauola/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V2_156BxLumiPileUp-v1/0007/52C1668C-4C7C-DE11-9ED5-000423D996C8.root',
       '/store/relval/CMSSW_3_2_2/RelValTTbar_Tauola/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V2_156BxLumiPileUp-v1/0007/525343E9-4E7C-DE11-BB53-000423D992A4.root',
       '/store/relval/CMSSW_3_2_2/RelValTTbar_Tauola/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V2_156BxLumiPileUp-v1/0007/4E79CA36-4E7C-DE11-B9F4-000423D98B6C.root',
       '/store/relval/CMSSW_3_2_2/RelValTTbar_Tauola/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V2_156BxLumiPileUp-v1/0007/4896D74B-4C7C-DE11-8C89-000423D95030.root',
       '/store/relval/CMSSW_3_2_2/RelValTTbar_Tauola/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V2_156BxLumiPileUp-v1/0007/46915BAA-4D7C-DE11-982C-000423D98B6C.root',
       '/store/relval/CMSSW_3_2_2/RelValTTbar_Tauola/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V2_156BxLumiPileUp-v1/0007/4266283B-517C-DE11-A330-001D09F29597.root',
       '/store/relval/CMSSW_3_2_2/RelValTTbar_Tauola/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V2_156BxLumiPileUp-v1/0007/2EE85113-5A7C-DE11-B311-000423D9853C.root',
       '/store/relval/CMSSW_3_2_2/RelValTTbar_Tauola/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V2_156BxLumiPileUp-v1/0007/2AD3566E-4F7C-DE11-919E-000423D98F98.root',
       '/store/relval/CMSSW_3_2_2/RelValTTbar_Tauola/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V2_156BxLumiPileUp-v1/0007/20C67138-777C-DE11-88DB-000423D6B5C4.root',
       '/store/relval/CMSSW_3_2_2/RelValTTbar_Tauola/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V2_156BxLumiPileUp-v1/0007/10E83CB6-587C-DE11-918C-000423D987FC.root',
       '/store/relval/CMSSW_3_2_2/RelValTTbar_Tauola/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V2_156BxLumiPileUp-v1/0007/0AD718DF-517C-DE11-A59E-000423DD2F34.root'
] )
