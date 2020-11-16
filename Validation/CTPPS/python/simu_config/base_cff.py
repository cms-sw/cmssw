import FWCore.ParameterSet.Config as cms

# load standard files
from RecoPPS.ProtonReconstruction.ctppsProtons_cff import *

# undo unapplicable settings
#del ctppsRPAlignmentCorrectionsDataESSourceXML
#del esPreferLocalAlignment

#ctppsOpticalFunctionsESSource.configuration = cms.VPSet()
#del ctppsOpticalFunctionsESSource
#del esPreferLocalOptics
del ctppsInterpolatedOpticalFunctionsESSource

# beam parameters as declared by LHC
ctppsLHCInfoESSource = cms.ESSource("CTPPSLHCInfoESSource",
  label = cms.string(""),
  validityRange = cms.EventRange("0:min - 999999:max"),
  beamEnergy = cms.double(6500),  # GeV
  xangle = cms.double(-1),  # murad
  betaStar = cms.double(-1)
)

# beam parameters as determined by PPS
ctppsBeamParametersESSource = cms.ESSource("CTPPSBeamParametersESSource",
  setBeamPars = cms.bool(True),

  #  beam momentum  (GeV)
  beamMom45 = cms.double(6500.),
  beamMom56 = cms.double(6500.),

  #  beta*  (cm)
  betaStarX45 = cms.double(0.),
  betaStarX56 = cms.double(0.),
  betaStarY45 = cms.double(0.),
  betaStarY56 = cms.double(0.),

  #  beam divergence  (rad)
  beamDivX45 = cms.double(30E-6),
  beamDivX56 = cms.double(30E-6),
  beamDivY45 = cms.double(30E-6),
  beamDivY56 = cms.double(30E-6),

  #  half crossing angle  (rad)
  halfXangleX45 = cms.double(-1),
  halfXangleX56 = cms.double(-1),
  halfXangleY45 = cms.double(0.),
  halfXangleY56 = cms.double(0.),

  #  vertex offset  (cm)
  vtxOffsetX45 = cms.double(0.),
  vtxOffsetX56 = cms.double(0.),
  vtxOffsetY45 = cms.double(0.),
  vtxOffsetY56 = cms.double(0.),
  vtxOffsetZ45 = cms.double(0.),
  vtxOffsetZ56 = cms.double(0.),
  vtxOffsetT45 = cms.double(0.),
  vtxOffsetT56 = cms.double(0.),

  #  vertex sigma  (cm)
  vtxStddevX = cms.double(10E-4),
  vtxStddevY = cms.double(10E-4),
  vtxStddevZ = cms.double(5),
  vtxStddevT = cms.double(6)
)

# particle-data table
from SimGeneral.HepPDTESSource.pythiapdt_cfi import *

# random seeds
RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
  sourceSeed = cms.PSet(initialSeed = cms.untracked.uint32(98765)),
  generator = cms.PSet(initialSeed = cms.untracked.uint32(98766)),
  beamDivergenceVtxGenerator = cms.PSet(initialSeed = cms.untracked.uint32(3849)),
  ctppsDirectProtonSimulation = cms.PSet(initialSeed = cms.untracked.uint32(4981))
)

# default source
source = cms.Source("EmptySource",
  firstRun = cms.untracked.uint32(1),
  numberEventsInLuminosityBlock = cms.untracked.uint32(10)
)

# particle generator
from Configuration.Generator.randomXiThetaGunProducer_cfi import *
generator.xi_max = 0.25
generator.theta_x_sigma = 60E-6
generator.theta_y_sigma = 60E-6

# beam smearing
from IOMC.EventVertexGenerators.beamDivergenceVtxGenerator_cfi import *

# direct simulation
from Validation.CTPPS.ctppsDirectProtonSimulation_cfi import *
ctppsDirectProtonSimulation.verbosity = 0
ctppsDirectProtonSimulation.hepMCTag = cms.InputTag('beamDivergenceVtxGenerator')
ctppsDirectProtonSimulation.roundToPitch = True
ctppsDirectProtonSimulation.pitchStrips = 66E-3 * 12 / 19 # effective value to reproduce real RP resolution
ctppsDirectProtonSimulation.pitchPixelsHor = 50E-3
ctppsDirectProtonSimulation.pitchPixelsVer = 80E-3
ctppsDirectProtonSimulation.produceHitsRelativeToBeam = True
ctppsDirectProtonSimulation.produceScoringPlaneHits = False
ctppsDirectProtonSimulation.produceRecHits = True

# local reconstruction
from RecoPPS.Local.totemRPLocalReconstruction_cff import *
from RecoPPS.Local.ctppsPixelLocalReconstruction_cff import *
from RecoPPS.Local.ctppsDiamondLocalReconstruction_cff import *
from RecoPPS.Local.ctppsLocalTrackLiteProducer_cff import *

totemRPUVPatternFinder.tagRecHit = cms.InputTag('ctppsDirectProtonSimulation')
ctppsPixelLocalTracks.label = "ctppsDirectProtonSimulation"
ctppsDiamondLocalTracks.recHitsTag = cms.InputTag('ctppsDirectProtonSimulation')

ctppsLocalTrackLiteProducer.includeDiamonds = False

# proton reconstruction
ctppsProtons.tagLocalTrackLite = cms.InputTag('ctppsLocalTrackLiteProducer')

#----------------------------------------------------------------------------------------------------

def SetSmearingLevel1(obj):
  obj.vtxStddevX = 0E-4
  obj.vtxStddevZ = 0

  obj.beamDivX45 = 0E-6
  obj.beamDivX56 = 0E-6
  obj.beamDivY45 = 0E-6
  obj.beamDivY56 = 0E-6

def SetLevel1(process):
  if hasattr(process, "ctppsBeamParametersESSource"):
    SetSmearingLevel1(process.ctppsBeamParametersESSource)
  else:
    SetSmearingLevel1(process.ctppsBeamParametersFromLHCInfoESSource)

  process.ctppsDirectProtonSimulation.roundToPitch = False

def SetSmearingLevel2(obj):
  obj.beamDivX45 = 0E-6
  obj.beamDivX56 = 0E-6
  obj.beamDivY45 = 0E-6
  obj.beamDivY56 = 0E-6

def SetLevel2(process):
  if hasattr(process, "ctppsBeamParametersESSource"):
    SetSmearingLevel2(process.ctppsBeamParametersESSource)
  else:
    SetSmearingLevel2(process.ctppsBeamParametersFromLHCInfoESSource)

  process.ctppsDirectProtonSimulation.roundToPitch = False


def SetLevel3(process):
  process.ctppsDirectProtonSimulation.roundToPitch = False


def SetLevel4(process):
  pass


def SetLowTheta(process):
  process.generator.theta_x_sigma = 0E-6
  process.generator.theta_y_sigma = 0E-6


def SetLargeTheta(process):
  pass

# xangle in murad
def UseCrossingAngle(xangle, process):
  process.ctppsLHCInfoESSource.xangle = xangle
  process.ctppsBeamParametersESSource.halfXangleX45 = xangle * 1E-6
  process.ctppsBeamParametersESSource.halfXangleX56 = xangle * 1E-6

default_xangle_beta_star_file = "CalibPPS/ESProducers/data/xangle_beta_distributions/version1.root"

def UseXangleBetaStarHistogram(process, f, obj):
  process.load("CalibPPS.ESProducers.ctppsLHCInfoRandomXangleESSource_cfi")
  process.ctppsLHCInfoRandomXangleESSource.generateEveryNEvents = 10 # this is to be synchronised with source.numberEventsInLuminosityBlock
  process.ctppsLHCInfoRandomXangleESSource.xangleBetaStarHistogramFile = f
  process.ctppsLHCInfoRandomXangleESSource.xangleBetaStarHistogramObject = obj
  process.ctppsLHCInfoRandomXangleESSource.beamEnergy = ctppsLHCInfoESSource.beamEnergy

  del process.ctppsLHCInfoESSource

  process.load("CalibPPS.ESProducers.ctppsBeamParametersFromLHCInfoESSource_cfi")
  process.ctppsBeamParametersFromLHCInfoESSource.beamDivX45 = process.ctppsBeamParametersESSource.beamDivX45
  process.ctppsBeamParametersFromLHCInfoESSource.beamDivX56 = process.ctppsBeamParametersESSource.beamDivX56
  process.ctppsBeamParametersFromLHCInfoESSource.beamDivY45 = process.ctppsBeamParametersESSource.beamDivY45
  process.ctppsBeamParametersFromLHCInfoESSource.beamDivY56 = process.ctppsBeamParametersESSource.beamDivY56
  process.ctppsBeamParametersFromLHCInfoESSource.vtxOffsetX45 = process.ctppsBeamParametersESSource.vtxOffsetX45
  process.ctppsBeamParametersFromLHCInfoESSource.vtxOffsetX56 = process.ctppsBeamParametersESSource.vtxOffsetX56
  process.ctppsBeamParametersFromLHCInfoESSource.vtxOffsetY45 = process.ctppsBeamParametersESSource.vtxOffsetY45
  process.ctppsBeamParametersFromLHCInfoESSource.vtxOffsetY56 = process.ctppsBeamParametersESSource.vtxOffsetY56
  process.ctppsBeamParametersFromLHCInfoESSource.vtxOffsetZ45 = process.ctppsBeamParametersESSource.vtxOffsetZ45
  process.ctppsBeamParametersFromLHCInfoESSource.vtxOffsetZ56 = process.ctppsBeamParametersESSource.vtxOffsetZ56
  process.ctppsBeamParametersFromLHCInfoESSource.vtxStddevX = process.ctppsBeamParametersESSource.vtxStddevX
  process.ctppsBeamParametersFromLHCInfoESSource.vtxStddevY = process.ctppsBeamParametersESSource.vtxStddevY
  process.ctppsBeamParametersFromLHCInfoESSource.vtxStddevZ = process.ctppsBeamParametersESSource.vtxStddevZ

  del process.ctppsBeamParametersESSource
