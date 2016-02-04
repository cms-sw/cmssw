import FWCore.ParameterSet.Config as cms

VertexAssociatorByTracksESProducer = cms.ESProducer("VertexAssociatorByTracksESProducer",
    # Matching conditions
    R2SMatchedSimRatio = cms.double(0.3),
    R2SMatchedRecoRatio = cms.double(0.0),
    S2RMatchedSimRatio = cms.double(0.0),
    S2RMatchedRecoRatio = cms.double(0.3),

    # RecoTrack selection
    trackQuality = cms.string("highPurity"),   

    # TrackingParticle selection
    trackingParticleSelector = cms.PSet(
        lipTP = cms.double(30.0),
        chargedOnlyTP = cms.bool(True),
        pdgIdTP = cms.vint32(),
        signalOnlyTP = cms.bool(True),
        minRapidityTP = cms.double(-2.4),
        minHitTP = cms.int32(0),
        ptMinTP = cms.double(0.9),
        maxRapidityTP = cms.double(2.4),
        tipTP = cms.double(3.5)
    )
)


