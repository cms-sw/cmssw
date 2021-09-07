import FWCore.ParameterSet.Config as cms

from Validation.CTPPS.simu_config.year_2022_cff import *

alignmentFile = "Validation/CTPPS/alignment/alignment_2022.xml"

profile_2022_default = profile_base_2022.clone(
  L_int = 1,

  ctppsLHCInfo = dict(
    # NB: until a dedicated 2022 distributions are issued, it is OK to use 2021 ones here
    xangleBetaStarHistogramObject = "2021/h2_betaStar_vs_xangle"
  ),

  ctppsRPAlignmentCorrectionsDataXML = dict(
    MisalignedFiles = [alignmentFile],
    RealFiles = [alignmentFile]
  ),

  ctppsDirectSimuData = dict(
    timeResolutionDiamonds45 = "0.200",
    timeResolutionDiamonds56 = "0.200"
  )
)
