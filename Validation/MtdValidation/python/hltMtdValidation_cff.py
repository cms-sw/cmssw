import FWCore.ParameterSet.Config as cms

# --- Cluster associations maps producers
from SimFastTiming.MtdAssociatorProducers.mtdRecoClusterToSimLayerClusterAssociatorByHits_cfi import mtdRecoClusterToSimLayerClusterAssociatorByHits as _mtdRecoClusterToSimLayerClusterAssociatorByHits
from SimFastTiming.MtdAssociatorProducers.mtdSimLayerClusterToTPAssociatorByTrackId_cfi import mtdSimLayerClusterToTPAssociatorByTrackId as _mtdSimLayerClusterToTPAssociatorByTrackId

from SimFastTiming.MtdAssociatorProducers.mtdRecoClusterToSimLayerClusterAssociation_cfi import mtdRecoClusterToSimLayerClusterAssociation as _mtdRecoClusterToSimLayerClusterAssociation
from SimFastTiming.MtdAssociatorProducers.mtdSimLayerClusterToTPAssociation_cfi import mtdSimLayerClusterToTPAssociation as _mtdSimLayerClusterToTPAssociation

hltMtdRecoClusterToSimLayerClusterAssociatorByHits = _mtdRecoClusterToSimLayerClusterAssociatorByHits.clone()
hltMtdSimLayerClusterToTPAssociatorByTrackId = _mtdSimLayerClusterToTPAssociatorByTrackId.clone()

hltMtdRecoClusterToSimLayerClusterAssociation =  _mtdRecoClusterToSimLayerClusterAssociation.clone(
    associator = cms.InputTag('hltMtdRecoClusterToSimLayerClusterAssociatorByHits'),
    btlRecoClustersTag = cms.InputTag('hltMtdClusters', 'FTLBarrel'),
    etlRecoClustersTag = cms.InputTag('hltMtdClusters', 'FTLEndcap'),
)

hltMtdSimLayerClusterToTPAssociation = _mtdSimLayerClusterToTPAssociation.clone(
    associator = cms.InputTag('hltMtdSimLayerClusterToTPAssociatorByTrackId')
)

hltMtdAssociationProducers = cms.Sequence(
    hltMtdRecoClusterToSimLayerClusterAssociatorByHits+
    hltMtdRecoClusterToSimLayerClusterAssociation+
    hltMtdSimLayerClusterToTPAssociatorByTrackId+
    hltMtdSimLayerClusterToTPAssociation
)

from Validation.MtdValidation.mtdTracksValid_cfi import mtdTracksValid as _mtdTracksValid
hltMtdTracksValid = _mtdTracksValid.clone(
    folder = cms.string('HLT/MTD/Tracks'),
    inputTagG = cms.InputTag('hltGeneralTracks'),
    inputTagT = cms.InputTag('hltTrackExtenderWithMTD'),
    inputTagV = cms.InputTag('hltOfflinePrimaryVertices4D'),
    TPtoRecoTrackAssoc = cms.InputTag('tpToHLTphase2TrackAssociation'),
    tp2SimAssociationMapTag = cms.InputTag('hltMtdSimLayerClusterToTPAssociation'),
    Sim2tpAssociationMapTag = cms.InputTag('hltMtdSimLayerClusterToTPAssociation'),
    r2sAssociationMapTag = cms.InputTag('hltMtdRecoClusterToSimLayerClusterAssociation'),
    btlRecHits = cms.InputTag('hltMtdRecHits', 'FTLBarrel'),
    etlRecHits = cms.InputTag('hltMtdRecHits', 'FTLEndcap'),
    recCluTagBTL = cms.InputTag('hltMtdClusters', 'FTLBarrel'),
    recCluTagETL = cms.InputTag('hltMtdClusters', 'FTLEndcap'),
    tmtd = cms.InputTag('hltTrackExtenderWithMTD', 'generalTracktmtd'),
    sigmatmtd = cms.InputTag('hltTrackExtenderWithMTD', 'generalTracksigmatmtd'),
    t0Src = cms.InputTag('hltTrackExtenderWithMTD', 'generalTrackt0'),
    sigmat0Src = cms.InputTag('hltTrackExtenderWithMTD', 'generalTracksigmat0'),
    trackAssocSrc = cms.InputTag('hltTrackExtenderWithMTD', 'generalTrackassoc'),
    pathLengthSrc = cms.InputTag('hltTrackExtenderWithMTD', 'generalTrackPathLength'),
    btlMatchTimeChi2 = cms.InputTag('hltTrackExtenderWithMTD', 'btlMatchTimeChi2'),
    etlMatchTimeChi2 = cms.InputTag('hltTrackExtenderWithMTD', 'etlMatchTimeChi2'),
    btlMatchChi2 = cms.InputTag('hltTrackExtenderWithMTD', 'btlMatchChi2'),
    t0SafePID = cms.InputTag('hltTofPID', 't0safe'),
    sigmat0SafePID = cms.InputTag('hltTofPID', 'sigmat0safe'),
    sigmat0PID = cms.InputTag('hltTofPID', 'sigmat0'),
    t0PID = cms.InputTag('hltTofPID', 't0'),
    sigmaTofPi = cms.InputTag('hltTrackExtenderWithMTD', 'generalTrackSigmaTofPi'),
    sigmaTofK = cms.InputTag('hltTrackExtenderWithMTD', 'generalTrackSigmaTofK'),
    sigmaTofP = cms.InputTag('hltTrackExtenderWithMTD', 'generalTrackSigmaTofP'),
    trackMVAQual = cms.InputTag('hltMtdTrackQualityMVA', 'mtdQualMVA'),
    outermostHitPositionSrc = cms.InputTag('hltTrackExtenderWithMTD', 'generalTrackOutermostHitPosition'),
)

from Validation.MtdValidation.vertices4DValid_cff import vertices4DValid as _vertices4DValid
hltVertices4DValid =  _vertices4DValid.clone(
    folder = cms.string('HLT/MTD/Vertices'),
    TPtoRecoTrackAssoc = cms.InputTag('tpToHLTphase2TrackAssociation'),
    TrackLabel = cms.InputTag('hltGeneralTracks'),
    mtdTracks = cms.InputTag('hltTrackExtenderWithMTD'),
    SimTag = cms.InputTag('mix', 'MergedTrackTruth'),
    offlineBS = cms.InputTag('hltOnlineBeamSpot'),
    offline4DPV = cms.InputTag('hltOfflinePrimaryVertices4D'),
    trackAssocSrc = cms.InputTag('hltTrackExtenderWithMTD', 'generalTrackassoc'),
    pathLengthSrc = cms.InputTag('hltTrackExtenderWithMTD', 'generalTrackPathLength'),
    momentumSrc = cms.InputTag('hltTrackExtenderWithMTD', 'generalTrackp'),
    tmtd = cms.InputTag('hltTrackExtenderWithMTD', 'generalTracktmtd'),
    timeSrc = cms.InputTag('hltTrackExtenderWithMTD', 'generalTracktmtd'),
    sigmaSrc = cms.InputTag('hltTrackExtenderWithMTD', 'generalTracksigmatmtd'),
    t0PID = cms.InputTag('hltTofPID', 't0'),
    sigmat0PID = cms.InputTag('hltTofPID', 'sigmat0'),
    t0SafePID = cms.InputTag('hltTofPID', 't0safe'),
    sigmat0SafePID = cms.InputTag('hltTofPID', 'sigmat0safe'),
    trackMVAQual = cms.InputTag('hltMtdTrackQualityMVA', 'mtdQualMVA'),
    tofPi = cms.InputTag('hltTrackExtenderWithMTD', 'generalTrackTofPi'),
    tofK = cms.InputTag('hltTrackExtenderWithMTD', 'generalTrackTofK'),
    tofP = cms.InputTag('hltTrackExtenderWithMTD', 'generalTrackTofP'),
    probPi = cms.InputTag('hltTofPID', 'probPi'),
    probK = cms.InputTag('hltTofPID', 'probK'),
    probP = cms.InputTag('hltTofPID', 'probP')
)

hltMtdRecoValid = cms.Sequence(#hltMtdAssociationProducers +
                               #hltMtdTracksValid +
                               hltVertices4DValid)
