import FWCore.ParameterSet.Config as cms

cutsCKF = cms.EDFilter("RecoTrackSelector",
    src = cms.InputTag("ctfWithMaterialTracks"),
    maxChi2 = cms.double(10000.0), ##not used in tdr

    tip = cms.double(120.0), ##3.5

    minRapidity = cms.double(-5.0), ##-2.5

    lip = cms.double(300.0), ##30

    #default cuts are dummy: cut used to produce TDR plots are commented
    ptMin = cms.double(0.1),
    maxRapidity = cms.double(5.0), ##2.5

    quality = cms.string('loose'),
    minHit = cms.int32(3) ##8

)


