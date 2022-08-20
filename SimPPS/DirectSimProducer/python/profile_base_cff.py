import FWCore.ParameterSet.Config as cms

profile_base = cms.PSet(
    L_int = cms.double(1.),
    # LHCInfo (default Run 3 values)
    ctppsLHCInfo = cms.PSet(
        xangle = cms.double(-1.),
        betaStar = cms.double(-1.),
        beamEnergy = cms.double(6.8e3),
        xangleBetaStarHistogramFile = cms.string("CalibPPS/ESProducers/data/xangle_beta_distributions/version1.root"),
        xangleBetaStarHistogramObject = cms.string("")
    ),
    # optics
    ctppsOpticalFunctions = cms.PSet(
        opticalFunctions = cms.VPSet(),
        scoringPlanes = cms.VPSet()
    ),
    # alignment
    ctppsRPAlignmentCorrectionsDataXML = cms.PSet(
        MeasuredFiles = cms.vstring(),
        RealFiles = cms.vstring(),
        MisalignedFiles = cms.vstring()
    ),
    # direct simu data
    ctppsDirectSimuData = cms.PSet(
        empiricalAperture45 = cms.string(""),
        empiricalAperture56 = cms.string(""),
        timeResolutionDiamonds45 = cms.string("999"),
        timeResolutionDiamonds56 = cms.string("999"),
        efficienciesPerRP = cms.VPSet(),
        efficienciesPerPlane = cms.VPSet()
    )
)
