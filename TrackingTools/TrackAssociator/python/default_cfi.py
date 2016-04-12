import FWCore.ParameterSet.Config as cms

# -*-SH-*-
#
# Default Parameters
#
#   Purpose: extraction of energy deposition and muon matching information
#            for a minimum ionizing particle. Up to 5x5 energy for ECAL 
#            and HCAL should be available.
# 
TrackAssociatorParameterBlock = cms.PSet(
    TrackAssociatorParameters = cms.PSet(
        muonMaxDistanceSigmaX = cms.double(0.0),
        muonMaxDistanceSigmaY = cms.double(0.0),
        CSCSegmentCollectionLabel = cms.InputTag("cscSegments"),        
        GEMSegmentCollectionLabel = cms.InputTag("gemSegments"),
        dRHcal = cms.double(9999.0),
        dREcal = cms.double(9999.0),
        CaloTowerCollectionLabel = cms.InputTag("towerMaker"),
        useEcal = cms.bool(True),
        dREcalPreselection = cms.double(0.05),
        HORecHitCollectionLabel = cms.InputTag("horeco"),
        dRMuon = cms.double(9999.0),
	trajectoryUncertaintyTolerance = cms.double(-1.0),
        propagateAllDirections = cms.bool(True),
        muonMaxDistanceX = cms.double(5.0),
        muonMaxDistanceY = cms.double(5.0),
        useHO = cms.bool(True),
        accountForTrajectoryChangeCalo = cms.bool(False),
        DTRecSegment4DCollectionLabel = cms.InputTag("dt4DSegments"),
        EERecHitCollectionLabel = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
        dRHcalPreselection = cms.double(0.2),
        useMuon = cms.bool(True),
        useCalo = cms.bool(False),
        EBRecHitCollectionLabel = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
        dRMuonPreselection = cms.double(0.2),
	usePreshower = cms.bool(False),
	dRPreshowerPreselection = cms.double(0.2),
        truthMatch = cms.bool(False),
        HBHERecHitCollectionLabel = cms.InputTag("hbhereco"),
        useHcal = cms.bool(True),
        maxPullXGE11 = cms.double (2.0),
        maxDiffXGE11 = cms.double (1.5),
        maxPullYGE11 = cms.double (2.0),
        maxDiffYGE11 = cms.double (10.0),
        maxPullXGE21 = cms.double (2.0),
        maxDiffXGE21 = cms.double (2.5),
        maxPullYGE21 = cms.double (2.0),
        maxDiffYGE21 = cms.double (12.0),
        maxDiffPhiDirection = cms.double (0.3),
    )
)
TrackAssociatorParameters = cms.PSet(
    muonMaxDistanceSigmaX = cms.double(0.0),
    muonMaxDistanceSigmaY = cms.double(0.0),
    CSCSegmentCollectionLabel = cms.InputTag("cscSegments"),
    GEMSegmentCollectionLabel = cms.InputTag("gemSegments"),
    dRHcal = cms.double(9999.0),
    dREcal = cms.double(9999.0),
    CaloTowerCollectionLabel = cms.InputTag("towerMaker"),
    useEcal = cms.bool(True),
    dREcalPreselection = cms.double(0.05),
    HORecHitCollectionLabel = cms.InputTag("horeco"),
    dRMuon = cms.double(9999.0),
    trajectoryUncertaintyTolerance = cms.double(-1.0),
    propagateAllDirections = cms.bool(True),
    muonMaxDistanceX = cms.double(5.0),
    muonMaxDistanceY = cms.double(5.0),
    useHO = cms.bool(True),
    accountForTrajectoryChangeCalo = cms.bool(False),
    DTRecSegment4DCollectionLabel = cms.InputTag("dt4DSegments"),
    EERecHitCollectionLabel = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
    dRMuonPreselection = cms.double(0.2),
    usePreshower = cms.bool(False),
    dRHcalPreselection = cms.double(0.2),
    useMuon = cms.bool(True),
    useCalo = cms.bool(False),
    EBRecHitCollectionLabel = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    truthMatch = cms.bool(False),
    HBHERecHitCollectionLabel = cms.InputTag("hbhereco"),
    useHcal = cms.bool(True)
)

## def _modifyRecoMuonPPonlyForPhase2( object ):
##     object.TrackAssociatorParameterBlock.TrackAssociatorParameters.GEMSegmentCollectionLabel = cms.InputTag("gemSegments")
##     object.TrackAssociatorParameters.GEMSegmentCollectionLabel = cms.InputTag("gemSegments")

## from Configuration.StandardSequences.Eras import eras
## modifyConfigurationStandardSequencesRecoMuonForPhase2_ = eras.phase2_muon.makeProcessModifier( _modifyRecoMuonPPonlyForPhase2 )

