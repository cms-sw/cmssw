import FWCore.ParameterSet.Config as cms

from Validation.CTPPS.simu_config.year_2018_cff import *

# alignment
from CalibPPS.ESProducers.ctppsRPAlignmentCorrectionsDataESSourceXML_cfi import *
alignmentFile = "Validation/CTPPS/alignment/2018_postTS2.xml"
ctppsRPAlignmentCorrectionsDataESSourceXML.MisalignedFiles = [alignmentFile]
ctppsRPAlignmentCorrectionsDataESSourceXML.RealFiles = [alignmentFile]

# timing resolution
ctppsDirectProtonSimulation.timeResolutionDiamonds45 = "2 * (-0.0031 * (x - 3) + 0.16)"
ctppsDirectProtonSimulation.timeResolutionDiamonds56 = "2 * ( (x<10)*(-0.0057*(x-10) + 0.110) + (x>=10)*(-0.0022*(x-10) + 0.110) )"

# xangle/beta* distribution
def UseXangleBetaStarDistribution(process, f):
  UseXangleBetaStarHistogram(process, f, "/2018_postTS2/h2_betaStar_vs_xangle")
