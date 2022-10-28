import FWCore.ParameterSet.Config as cms
from SimPPS.DirectSimProducer.profile_base_cff import profile_base as _base
from CalibPPS.ESProducers.ctppsOpticalFunctions_non_DB_cff import optics_2021 as _optics

# base profile settings for 2022
_base_2022 = _base.clone(
    ctppsOpticalFunctions = _base.ctppsOpticalFunctions.clone(
        opticalFunctions = _optics.opticalFunctions,
        scoringPlanes = _optics.scoringPlanes,
    ),
    ctppsDirectSimuData = _base.ctppsDirectSimuData.clone(
        empiricalAperture45 = "1.e3*([xi] - 0.20)",
        empiricalAperture56 = "1.e3*([xi] - 0.20)"
    )
)

profile_2022_default = _base_2022.clone(
    L_int = 1.,
    ctppsLHCInfo = _base_2022.ctppsLHCInfo.clone(
        # NB: until a dedicated 2022 distributions are issued, it is OK to use 2021 ones here
        xangleBetaStarHistogramObject = "2021/h2_betaStar_vs_xangle"
    ),
    ctppsRPAlignmentCorrectionsDataXML = _base_2022.ctppsRPAlignmentCorrectionsDataXML.clone(
        MisalignedFiles = ["Validation/CTPPS/alignment/alignment_2022.xml"],
        RealFiles = ["Validation/CTPPS/alignment/alignment_2022.xml"]
    ),
    ctppsDirectSimuData = _base_2022.ctppsDirectSimuData.clone(
        timeResolutionDiamonds45 = "0.200",
        timeResolutionDiamonds56 = "0.200"
    )
)
