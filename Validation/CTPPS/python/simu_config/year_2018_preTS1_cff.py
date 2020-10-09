import FWCore.ParameterSet.Config as cms

from Validation.CTPPS.simu_config.year_2018_cff import *

# alignment
from CalibPPS.ESProducers.ctppsRPAlignmentCorrectionsDataESSourceXML_cfi import *
alignmentFile = "Validation/CTPPS/alignment/2018_preTS1.xml"
ctppsRPAlignmentCorrectionsDataESSourceXML.MisalignedFiles = [alignmentFile]
ctppsRPAlignmentCorrectionsDataESSourceXML.RealFiles = [alignmentFile]

# timing not available in this period
ctppsLocalTrackLiteProducer.includeDiamonds = False
ctppsDirectProtonSimulation.timeResolutionDiamonds45 = "999"
ctppsDirectProtonSimulation.timeResolutionDiamonds56 = "999"

# xangle/beta* options
def UseDefaultXangleBetaStar(process):
  UseCrossingAngle(140, process)

def UseDefaultXangleBetaStarDistribution(process):
  UseXangleBetaStarHistogram(process, default_xangle_beta_star_file, "2018_preTS1/h2_betaStar_vs_xangle")

# defaults
def SetDefaults(process):
  UseDefaultXangleBetaStarDistribution(process)
