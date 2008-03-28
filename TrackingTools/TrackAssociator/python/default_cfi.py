import FWCore.ParameterSet.Config as cms

# -*-SH-*-
#
# Default Parameters
#
#   Purpose: extraction of energy deposition and muon matching information
#            for a minimum ionizing particle. Up to 5x5 energy for ECAL 
#            and HCAL should be available.
# 
TrackAssociatorParameters = cms.PSet(
    muonMaxDistanceSigmaX = cms.double(0.0),
    muonMaxDistanceSigmaY = cms.double(0.0),
    CSCSegmentCollectionLabel = cms.InputTag("cscSegments"),
    dRHcal = cms.double(9999.0),
    # bool   accountForTrajectoryChangeMuon = true
    # matching requirements 
    dREcal = cms.double(9999.0),
    CaloTowerCollectionLabel = cms.InputTag("towerMaker"),
    # association types
    useEcal = cms.bool(True),
    # preselection requirements in theta-phi space
    # allowed range: 
    #   dTheta = +/- dR
    #   dPhi = +/- dR  
    # (track trajectory changes are taken into account for muons)
    dREcalPreselection = cms.double(0.05),
    HORecHitCollectionLabel = cms.InputTag("horeco"),
    dRMuon = cms.double(9999.0),
    crossedEnergyType = cms.string('SinglePointAlongTrajectory'),
    muonMaxDistanceX = cms.double(5.0),
    muonMaxDistanceY = cms.double(5.0),
    useHO = cms.bool(True), ## RecoHits

    accountForTrajectoryChangeCalo = cms.bool(False),
    DTRecSegment4DCollectionLabel = cms.InputTag("dt4DSegments"),
    EERecHitCollectionLabel = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
    dRHcalPreselection = cms.double(0.2),
    useMuon = cms.bool(True), ## RecoHits

    useCalo = cms.bool(False), ## CaloTowers

    # input tags
    EBRecHitCollectionLabel = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    dRMuonPreselection = cms.double(0.2),
    truthMatch = cms.bool(False), ## debugging information

    HBHERecHitCollectionLabel = cms.InputTag("hbhereco"),
    useHcal = cms.bool(True) ## RecoHits

)

