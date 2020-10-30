import FWCore.ParameterSet.Config as cms

from Validation.CTPPS.simu_config.year_2018_cff import *
profile_2018_postTS2=profile.clone()

#LHCInfo
profile_2018_postTS2.ctppsLHCInfo.xangleBetaStarHistogramObject="2018_postTS2/h2_betaStar_vs_xangle"
# alignment
alignmentFile = "Validation/CTPPS/alignment/2018_postTS2.xml"
profile_2018_postTS2.ctppsRPAlignmentCorrectionsDataXML.MisalignedFiles = [alignmentFile]
profile_2018_postTS2.ctppsRPAlignmentCorrectionsDataXML.RealFiles = [alignmentFile]

profile_2018_postTS2.ctppsDirectSimuData.timeResolutionDiamonds45="2*((x<16)*(-0.381504+0.255095*x-0.0415622*x^2+0.00275877*x^3-6.47115E-05*x^4)+(x>=16)*0.118)"
profile_2018_postTS2.ctppsDirectSimuData.timeResolutionDiamonds56="2*((x<16)*(-0.279298+0.219838*x-0.0384257*x^2+0.00268906*x^3-6.60572E-05*x^4)+(x>=16)*0.099)"

# time det efficiency
profile_2018_postTS2.ctppsDirectSimuData.useTimeEfficiencyCheck = False
profile_2018_postTS2.ctppsDirectSimuData.effTimePath = ""
profile_2018_postTS2.ctppsDirectSimuData.effTimeObject45="eff45"
profile_2018_postTS2.ctppsDirectSimuData.effTimeObject56="eff56"

