import FWCore.ParameterSet.Config as cms
from SimPPS.DirectSimProducer.profile_base_cff import profile_base as _base
from CalibPPS.ESProducers.ctppsOpticalFunctions_non_DB_cff import optics_2018 as _optics

# base profile settings for 2018
_base_2018 = _base.clone(
    ctppsLHCInfo = _base.ctppsLHCInfo.clone(
        beamEnergy = 6500.
    ),
    ctppsOpticalFunctions = _base.ctppsOpticalFunctions.clone(
        opticalFunctions = _optics.opticalFunctions,
        scoringPlanes = _optics.scoringPlanes,
    ),
    ctppsDirectSimuData = _base.ctppsDirectSimuData.clone(
        empiricalAperture45 = "-(8.44219E-07*[xangle]-0.000100957)+(([xi]<(0.000247185*[xangle]+0.101599))*-(1.40289E-05*[xangle]-0.00727237)+([xi]> = (0.000247185*[xangle]+0.101599))*-(0.000107811*[xangle]-0.0261867))*([xi]-(0.000247185*[xangle]+0.101599))",
        empiricalAperture56 = "-(-4.74758E-07*[xangle]+3.0881E-05)+(([xi]<(0.000727859*[xangle]+0.0722653))*-(2.43968E-05*[xangle]-0.0085461)+([xi]> = (0.000727859*[xangle]+0.0722653))*-(7.19216E-05*[xangle]-0.0148267))*([xi]-(0.000727859*[xangle]+0.0722653))"
    )
)

profile_2018_preTS1 = _base_2018.clone(
    L_int = 18.488297964,
    ctppsLHCInfo = _base_2018.ctppsLHCInfo.clone(
        xangleBetaStarHistogramObject = "2018_preTS1/h2_betaStar_vs_xangle"
    ),
    ctppsRPAlignmentCorrectionsDataXML = _base_2018.ctppsRPAlignmentCorrectionsDataXML.clone(
        MisalignedFiles = ["Validation/CTPPS/alignment/2018_preTS1.xml"],
        RealFiles = ["Validation/CTPPS/alignment/2018_preTS1.xml"]
    ),
    ctppsDirectSimuData = _base_2018.ctppsDirectSimuData.clone(
        # timing not available in this period
        timeResolutionDiamonds45 = "0.200",
        timeResolutionDiamonds56 = "0.200"
    )
)

profile_2018_TS1_TS2 = _base_2018.clone(
    L_int = 26.812002394,
    ctppsLHCInfo = _base_2018.ctppsLHCInfo.clone(
        xangleBetaStarHistogramObject = "2018_TS1_TS2/h2_betaStar_vs_xangle"
    ),
    ctppsRPAlignmentCorrectionsDataXML = _base_2018.ctppsRPAlignmentCorrectionsDataXML.clone(
        MisalignedFiles = ["Validation/CTPPS/alignment/2018_TS1_TS2.xml"],
        RealFiles = ["Validation/CTPPS/alignment/2018_TS1_TS2.xml"]
    ),
    ctppsDirectSimuData = _base_2018.ctppsDirectSimuData.clone(
        timeResolutionDiamonds45 = "2*((x<16)*(-0.171784+0.175856*x-0.0322344*x^2+0.00231489*x^3-5.7575E-05*x^4)+(x>=16)*0.105)",
        timeResolutionDiamonds56 = "2*((x<16)*(-0.014943+0.102806*x-0.0209404*x^2+0.00158264*x^3-4.08241E-05*x^4)+(x>=16)*0.089)"
    )
)

profile_2018_postTS2 = _base_2018.clone(
    L_int = 10.415769561,
    ctppsLHCInfo = _base_2018.ctppsLHCInfo.clone(
        xangleBetaStarHistogramObject = "2018_postTS2/h2_betaStar_vs_xangle"
    ),
    ctppsRPAlignmentCorrectionsDataXML = _base_2018.ctppsRPAlignmentCorrectionsDataXML.clone(
        MisalignedFiles = ["Validation/CTPPS/alignment/2018_postTS2.xml"],
        RealFiles = ["Validation/CTPPS/alignment/2018_postTS2.xml"]
    ),
    ctppsDirectSimuData = _base_2018.ctppsDirectSimuData.clone(
        timeResolutionDiamonds45 = "2*((x<16)*(-0.381504+0.255095*x-0.0415622*x^2+0.00275877*x^3-6.47115E-05*x^4)+(x>=16)*0.118)",
        timeResolutionDiamonds56 = "2*((x<16)*(-0.279298+0.219838*x-0.0384257*x^2+0.00268906*x^3-6.60572E-05*x^4)+(x>=16)*0.099)",
    )
)
