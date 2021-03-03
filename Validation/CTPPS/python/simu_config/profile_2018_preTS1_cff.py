import FWCore.ParameterSet.Config as cms

from Validation.CTPPS.simu_config.year_2018_cff import *

alignmentFile = "Validation/CTPPS/alignment/2018_preTS1.xml"

profile_2018_preTS1 = profile_base_2018.clone(
  L_int = 18.488297964,

  ctppsLHCInfo = dict(
    xangleBetaStarHistogramObject = "2018_preTS1/h2_betaStar_vs_xangle"
  ),

  ctppsRPAlignmentCorrectionsDataXML = dict(
    MisalignedFiles = [alignmentFile],
    RealFiles = [alignmentFile]
  ),

  ctppsDirectSimuData = dict(
    # timing not available in this period
    timeResolutionDiamonds45 = "0.200",
    timeResolutionDiamonds56 = "0.200"
  )
)
