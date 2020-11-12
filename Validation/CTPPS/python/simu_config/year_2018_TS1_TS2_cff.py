import FWCore.ParameterSet.Config as cms

from Validation.CTPPS.simu_config.year_2018_cff import *

# alignment
from CalibPPS.ESProducers.ctppsRPAlignmentCorrectionsDataESSourceXML_cfi import *
alignmentFile = "Validation/CTPPS/alignment/2018_TS1_TS2.xml"
ctppsRPAlignmentCorrectionsDataESSourceXML.MisalignedFiles = [alignmentFile]
ctppsRPAlignmentCorrectionsDataESSourceXML.RealFiles = [alignmentFile]

# timing resolution
ctppsDirectProtonSimulation.timeResolutionDiamonds45 = "2 * ( (x<10)*(-0.0086*(x-10) + 0.100) + (x>=10)*(0.100) )"
ctppsDirectProtonSimulation.timeResolutionDiamonds56 = "2 * ( (x<8) *(-0.0100*(x-8)  + 0.100) + (x>=8) *(-0.0027*(x-8) + 0.100) )"

# xangle/beta* options
def UseDefaultXangleBetaStar(process):
  UseCrossingAngle(140, process)

def UseDefaultXangleBetaStarDistribution(process):
  UseXangleBetaStarHistogram(process, default_xangle_beta_star_file, "2018_TS1_TS2/h2_betaStar_vs_xangle")

# defaults
def SetDefaults(process):
  UseDefaultXangleBetaStarDistribution(process)
