import FWCore.ParameterSet.Config as cms
from SimPPS.DirectSimProducer.profile_base_cff import profile_base as _base
from CalibPPS.ESProducers.ctppsOpticalFunctions_non_DB_cff import optics_2017 as _optics

# base profile settings for 2017
_base_2017 = _base.clone(
    ctppsLHCInfo = _base.ctppsLHCInfo.clone(
        beamEnergy = 6500.
    ),
    ctppsOpticalFunctions = _base.ctppsOpticalFunctions.clone(
        opticalFunctions = _optics.opticalFunctions,
        scoringPlanes = _optics.scoringPlanes,
    )
)

profile_2017_preTS2 = _base_2017.clone(
    L_int = 15.012899190,
    ctppsLHCInfo = _base_2017.ctppsLHCInfo.clone(
        xangleBetaStarHistogramObject = "2017_preTS2/h2_betaStar_vs_xangle"
    ),
    ctppsRPAlignmentCorrectionsDataXML = _base_2017.ctppsRPAlignmentCorrectionsDataXML.clone(
        MisalignedFiles = ["Validation/CTPPS/alignment/2017_preTS2.xml"],
        RealFiles = ["Validation/CTPPS/alignment/2017_preTS2.xml"]
    ),
    ctppsDirectSimuData = _base_2017.ctppsDirectSimuData.clone(
        empiricalAperture45 = "-(8.71198E-07*[xangle]-0.000134726)+(([xi]<(0.000264704*[xangle]+0.081951))*-(4.32065E-05*[xangle]-0.0130746)+([xi]>=(0.000264704*[xangle]+0.081951))*-(0.000183472*[xangle]-0.0395241))*([xi]-(0.000264704*[xangle]+0.081951))",
        empiricalAperture56 = "3.43116E-05+(([xi]<(0.000626936*[xangle]+0.061324))*0.00654394+([xi]>=(0.000626936*[xangle]+0.061324))*-(0.000145164*[xangle]-0.0272919))*([xi]-(0.000626936*[xangle]+0.061324))",
        timeResolutionDiamonds45 = "2*(-0.10784+0.105194*x-0.0182611*x^2+0.00134731*x^3-3.58212E-05*x^4)",
        timeResolutionDiamonds56 = "2*(0.00735552+0.0272707*x-0.00247151*x^2+8.62788E-05*x^3-7.99605E-07*x^4)"
    )
)

profile_2017_postTS2 = _base_2017.clone(
    L_int = 22.179613387,
    ctppsLHCInfo = _base_2017.ctppsLHCInfo.clone(
        xangleBetaStarHistogramObject = "2017_postTS2/h2_betaStar_vs_xangle"
    ),
    ctppsRPAlignmentCorrectionsDataXML = _base_2017.ctppsRPAlignmentCorrectionsDataXML.clone(
        MisalignedFiles = ["Validation/CTPPS/alignment/2017_postTS2.xml"],
        RealFiles = ["Validation/CTPPS/alignment/2017_postTS2.xml"]
    ),
    ctppsDirectSimuData = _base_2017.ctppsDirectSimuData.clone(
        empiricalAperture45 = "-(8.92079E-07*[xangle]-0.000150214)+(([xi]<(0.000278622*[xangle]+0.0964383))*-(3.9541e-05*[xangle]-0.0115104)+([xi]>=(0.000278622*[xangle]+0.0964383))*-(0.000108249*[xangle]-0.0249303))*([xi]-(0.000278622*[xangle]+0.0964383))",
        empiricalAperture56 = "4.56961E-05+(([xi]<(0.00075625*[xangle]+0.0643361))*-(3.01107e-05*[xangle]-0.00985126)+([xi]>=(0.00075625*[xangle]+0.0643361))*-(8.95437e-05*[xangle]-0.0169474))*([xi]-(0.00075625*[xangle]+0.0643361))",
        timeResolutionDiamonds45 = "2*(0.0152613+0.0498784*x-0.00824168*x^2+0.000599844*x^3-1.5923E-05*x^4)",
        timeResolutionDiamonds56 = "2*(-0.00458856+0.0522619*x-0.00806666*x^2+0.000558331*x^3-1.42165E-05*x^4)"
    )
)
