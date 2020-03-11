import FWCore.ParameterSet.Config as cms

# load standard files
from RecoCTPPS.ProtonReconstruction.ctppsProtons_cff import *

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
  xangle = cms.double(-1)  # murad
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

  #  vertex sigma  (cm)
  vtxStddevX = cms.double(10E-4),
  vtxStddevY = cms.double(10E-4),
  vtxStddevZ = cms.double(5)
)

# particle-data table
from SimGeneral.HepPDTESSource.pythiapdt_cfi import *

# random seeds
RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
  sourceSeed = cms.PSet(initialSeed =cms.untracked.uint32(98765)),
  generator = cms.PSet(initialSeed = cms.untracked.uint32(98766)),
  beamDivergenceVtxGenerator = cms.PSet(initialSeed =cms.untracked.uint32(3849))
)

# default source
source = cms.Source("EmptySource",
  firstRun = cms.untracked.uint32(1)
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
from RecoCTPPS.TotemRPLocal.totemRPLocalReconstruction_cff import *
from RecoCTPPS.PixelLocal.ctppsPixelLocalReconstruction_cff import *
from RecoCTPPS.TotemRPLocal.ctppsLocalTrackLiteProducer_cff import *

totemRPUVPatternFinder.tagRecHit = cms.InputTag('ctppsDirectProtonSimulation')
ctppsPixelLocalTracks.label = "ctppsDirectProtonSimulation"
ctppsLocalTrackLiteProducer.includeDiamonds = False

# proton reconstruction
ctppsProtons.tagLocalTrackLite = cms.InputTag('ctppsLocalTrackLiteProducer')

#----------------------------------------------------------------------------------------------------

def SetLevel1(process):
  process.ctppsBeamParametersESSource.vtxStddevX = 0E-4
  process.ctppsBeamParametersESSource.vtxStddevZ = 0

  process.ctppsBeamParametersESSource.beamDivX45 = 0E-6
  process.ctppsBeamParametersESSource.beamDivX56 = 0E-6
  process.ctppsBeamParametersESSource.beamDivY45 = 0E-6
  process.ctppsBeamParametersESSource.beamDivY56 = 0E-6

  process.ctppsDirectProtonSimulation.roundToPitch = False


def SetLevel2(process):
  process.ctppsBeamParametersESSource.beamDivX45 = 0E-6
  process.ctppsBeamParametersESSource.beamDivX56 = 0E-6
  process.ctppsBeamParametersESSource.beamDivY45 = 0E-6
  process.ctppsBeamParametersESSource.beamDivY56 = 0E-6

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
