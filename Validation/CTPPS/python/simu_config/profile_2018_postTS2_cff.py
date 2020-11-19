import FWCore.ParameterSet.Config as cms

from Validation.CTPPS.simu_config.year_2018_cff import *

alignmentFile = "Validation/CTPPS/alignment/2018_postTS2.xml"

profile_2018_postTS2 = profile_base_2018.clone(
  L_int = 10.415769561,

  ctppsLHCInfo = dict(
    xangleBetaStarHistogramObject = "2018_postTS2/h2_betaStar_vs_xangle"
  ),

  ctppsRPAlignmentCorrectionsDataXML = dict(
    MisalignedFiles = [alignmentFile],
    RealFiles = [alignmentFile]
  ),

  ctppsDirectSimuData = dict(
    timeResolutionDiamonds45 = "2*((x<16)*(-0.381504+0.255095*x-0.0415622*x^2+0.00275877*x^3-6.47115E-05*x^4)+(x>=16)*0.118)",
    timeResolutionDiamonds56 = "2*((x<16)*(-0.279298+0.219838*x-0.0384257*x^2+0.00268906*x^3-6.60572E-05*x^4)+(x>=16)*0.099)",
  )
)
