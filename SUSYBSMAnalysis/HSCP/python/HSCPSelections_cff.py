
import FWCore.ParameterSet.Config as cms

HSCPSelectionEmpty = cms.PSet(
      cms.PSet(
         onlyConsiderTrack        = cms.bool(False),
         onlyConsiderMuon         = cms.bool(False),
         onlyConsiderMuonSTA      = cms.bool(False),
         onlyConsiderMuonGB       = cms.bool(False),
         onlyConsiderMuonTK       = cms.bool(False),
         onlyConsiderRpc          = cms.bool(False),
         onlyConsiderEcal         = cms.bool(False),

         minTrackHits             = cms.int32(0),
         minTrackP                = cms.double(0),
         minTrackPt               = cms.double(0),

         minDedxEstimator         = cms.double(0),
         minDedxDiscriminator     = cms.double(0),

         minMuonP                 = cms.double(0),
         minMuonPt                = cms.double(0),

         maxMuTimeDtBeta          = cms.double(1),
         minMuTimeDtNdof          = cms.double(0),
         maxMuTimeCscBeta         = cms.double(1),
         minMuTimeCscNdof         = cms.double(0),
         maxMuTimeCombinedBeta    = cms.double(1),
         minMuTimeCombinedNdof    = cms.double(0),

         maxBetaRpc               = cms.double(0),
         maxBetaEcal              = cms.double(0),
      ),
)


HSCPSelectionDefault = HSCPSelectionEmpty.clone()
HSCPSelectionDefault.minTrackHits             = cms.int32(3)
HSCPSelectionDefault.minTrackPt               = cms.double(5)
HSCPSelectionDefault.minMuonPt                = cms.double(5)

HSCPSelectionHighdEdx = HSCPSelectionDefault.clone()
HSCPSelectionHighdEdx.onlyConsiderTrack       = cms.bool(True)
HSCPSelectionHighdEdx.minDedxEstimator        = cms.double(3.5)

HSCPSelectionHighTOF = HSCPSelectionDefault.clone()
HSCPSelectionHighTOF.onlyConsiderMuon         = cms.bool(True)
HSCPSelectionHighTOF.maxMuTimeDtBeta          = cms.double(0.9)
