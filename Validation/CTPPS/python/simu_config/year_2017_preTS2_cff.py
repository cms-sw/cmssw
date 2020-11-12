import FWCore.ParameterSet.Config as cms

from Validation.CTPPS.simu_config.year_2017_cff import *

# alignment
from CalibPPS.ESProducers.ctppsRPAlignmentCorrectionsDataESSourceXML_cfi import *
alignmentFile = "Validation/CTPPS/alignment/2017_preTS2.xml"
ctppsRPAlignmentCorrectionsDataESSourceXML.MisalignedFiles = [alignmentFile]
ctppsRPAlignmentCorrectionsDataESSourceXML.RealFiles = [alignmentFile]

# aperture cuts
ctppsDirectProtonSimulation.useEmpiricalApertures = True

ctppsDirectProtonSimulation.empiricalAperture45="-(8.71198E-07*[xangle]-0.000134726)+(([xi]<(0.000264704*[xangle]+0.081951))*-(4.32065E-05*[xangle]-0.0130746)+([xi]>=(0.000264704*[xangle]+0.081951))*-(0.000183472*[xangle]-0.0395241))*([xi]-(0.000264704*[xangle]+0.081951))"
ctppsDirectProtonSimulation.empiricalAperture56="3.43116E-05+(([xi]<(0.000626936*[xangle]+0.061324))*0.00654394+([xi]>=(0.000626936*[xangle]+0.061324))*-(0.000145164*[xangle]-0.0272919))*([xi]-(0.000626936*[xangle]+0.061324))"

# timing resolution
ctppsDirectProtonSimulation.timeResolutionDiamonds45 = "2 * (0.0025*(x-3) + 0.080)"
ctppsDirectProtonSimulation.timeResolutionDiamonds56 = "2 * (0.0050*(x-3) + 0.060)"

# xangle/beta* options
def UseDefaultXangleBetaStar(process):
  UseCrossingAngle(140, process)

def UseDefaultXangleBetaStarDistribution(process):
  UseXangleBetaStarHistogram(process, default_xangle_beta_star_file, "2017_preTS2/h2_betaStar_vs_xangle")

# defaults
def SetDefaults(process):
  UseDefaultXangleBetaStarDistribution(process)
