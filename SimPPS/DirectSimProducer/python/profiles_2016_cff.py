import FWCore.ParameterSet.Config as cms
from SimPPS.DirectSimProducer.profile_base_cff import profile_base as _base
from CalibPPS.ESProducers.ctppsOpticalFunctions_non_DB_cff import optics_2016_preTS2, optics_2016_postTS2

# base profile settings for 2016
_base_2016 = _base.clone(
    ctppsLHCInfo = _base.ctppsLHCInfo.clone(
        beamEnergy = 6500.
    )
)

profile_2016_preTS2 = _base_2016.clone(
    L_int = 6.138092276 + 3.654039035,
    ctppsLHCInfo = _base_2016.ctppsLHCInfo.clone(
        xangleBetaStarHistogramObject = "2016_preTS2/h2_betaStar_vs_xangle"
    ),
    ctppsOpticalFunctions = _base_2016.ctppsOpticalFunctions.clone(
        opticalFunctions = optics_2016_preTS2.opticalFunctions,
        scoringPlanes = optics_2016_preTS2.scoringPlanes,
    ),
    ctppsRPAlignmentCorrectionsDataXML = _base_2016.ctppsRPAlignmentCorrectionsDataXML.clone(
        MisalignedFiles = ["Validation/CTPPS/alignment/2016_preTS2.xml"],
        RealFiles = ["Validation/CTPPS/alignment/2016_preTS2.xml"]
    ),
    ctppsDirectSimuData = _base_2016.ctppsDirectSimuData.clone(
        empiricalAperture45 = "3.76296E-05+(([xi]<0.117122)*0.00712775+([xi]>=0.117122)*0.0148651)*([xi]-0.117122)",
        empiricalAperture56 = "1.85954E-05+(([xi]<0.14324)*0.00475349+([xi]>=0.14324)*0.00629514)*([xi]-0.14324)"
    )
)

profile_2016_postTS2 = _base_2016.clone(
    L_int = 5.007365807,
    ctppsLHCInfo = _base_2016.ctppsLHCInfo.clone(
        xangleBetaStarHistogramObject = "2016_postTS2/h2_betaStar_vs_xangle"
    ),
    ctppsOpticalFunctions = _base_2016.ctppsOpticalFunctions.clone(
        opticalFunctions = optics_2016_postTS2.opticalFunctions,
        scoringPlanes = optics_2016_postTS2.scoringPlanes,
    ),
    ctppsRPAlignmentCorrectionsDataXML = _base_2016.ctppsRPAlignmentCorrectionsDataXML.clone(
        MisalignedFiles = ["Validation/CTPPS/alignment/2016_postTS2.xml"],
        RealFiles = ["Validation/CTPPS/alignment/2016_postTS2.xml"]
    ),
    # direct simu data
    ctppsDirectSimuData = _base_2016.ctppsDirectSimuData.clone(
        empiricalAperture45 = "6.10374E-05+(([xi]<0.113491)*0.00795942+([xi]>=0.113491)*0.01935)*([xi]-0.113491)",
        empiricalAperture56 = "([xi]-0.110)/130.0"
    )
)
