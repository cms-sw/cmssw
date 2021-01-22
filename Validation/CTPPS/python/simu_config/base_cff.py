import FWCore.ParameterSet.Config as cms

# load standard files (on top so as settings can be overwritten below)
from RecoPPS.ProtonReconstruction.ctppsProtons_cff import *

# configuration for composite source of alignment, optics, ...
from CalibPPS.ESProducers.ctppsCompositeESSource_cfi import *
ctppsCompositeESSource.generateEveryNEvents = 100

profile_base = cms.PSet(
  L_int = cms.double(1),

  # LHCInfo
  ctppsLHCInfo = cms.PSet(
    xangle = cms.double(-1),
    betaStar = cms.double(-1),
  	beamEnergy  =  cms.double(0),
  	xangleBetaStarHistogramFile = cms.string("CalibPPS/ESProducers/data/xangle_beta_distributions/version1.root"),
  	xangleBetaStarHistogramObject = cms.string("")
  ),

  # optics
  ctppsOpticalFunctions = cms.PSet(
  	opticalFunctions = cms.VPSet(),
  	scoringPlanes = cms.VPSet()
  ),

  # alignment
  ctppsRPAlignmentCorrectionsDataXML = cms.PSet(
    MeasuredFiles = cms.vstring(),
    RealFiles = cms.vstring(),
    MisalignedFiles = cms.vstring()
  ),

  # direct simu data
  ctppsDirectSimuData = cms.PSet(
    empiricalAperture45 = cms.string(""),
    empiricalAperture56 = cms.string(""),

    timeResolutionDiamonds45 = cms.string("999"),
    timeResolutionDiamonds56 = cms.string("999"),

    effTimePath = cms.string(""),
    effTimeObject45 = cms.string(""),
    effTimeObject56 = cms.string("")
  )
)

# beam parameters as determined by PPS
ctppsBeamParametersFromLHCInfoESSource = cms.ESProducer("CTPPSBeamParametersFromLHCInfoESSource",
  lhcInfoLabel = cms.string(""),

  #  beam divergence  (rad)
  beamDivX45 = cms.double(30E-6),
  beamDivX56 = cms.double(30E-6),
  beamDivY45 = cms.double(30E-6),
  beamDivY56 = cms.double(30E-6),

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
  sourceSeed = cms.PSet(initialSeed = cms.untracked.uint32(98765)),
  generator = cms.PSet(initialSeed = cms.untracked.uint32(98766)),
  beamDivergenceVtxGenerator = cms.PSet(initialSeed = cms.untracked.uint32(3849)),
  ctppsDirectProtonSimulation = cms.PSet(initialSeed = cms.untracked.uint32(4981))
)

# default source
source = cms.Source("EmptySource",
  firstRun = cms.untracked.uint32(1),
  numberEventsInLuminosityBlock = ctppsCompositeESSource.generateEveryNEvents
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
ctppsDirectProtonSimulation.useEmpiricalApertures = True
ctppsDirectProtonSimulation.useTimingRPEfficiency = False
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
# utility functions

def SetSmearingLevel1(obj):
  obj.vtxStddevX = 0E-4
  obj.vtxStddevZ = 0

  obj.beamDivX45 = 0E-6
  obj.beamDivX56 = 0E-6
  obj.beamDivY45 = 0E-6
  obj.beamDivY56 = 0E-6

def SetLevel1(process):
  SetSmearingLevel1(process.ctppsBeamParametersFromLHCInfoESSource)

  process.ctppsDirectProtonSimulation.roundToPitch = False

def SetSmearingLevel2(obj):
  obj.beamDivX45 = 0E-6
  obj.beamDivX56 = 0E-6
  obj.beamDivY45 = 0E-6
  obj.beamDivY56 = 0E-6

def SetLevel2(process):
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

def UseConstantXangleBetaStar(process, xangle, betaStar):
  for p in ctppsCompositeESSource.periods:
    p.ctppsLHCInfo.xangle = xangle
    p.ctppsLHCInfo.betaStar = betaStar

def UseXangleBetaStarHistogram(process, f="", obj=""):
  for p in ctppsCompositeESSource.periods:
    p.ctppsLHCInfo.xangle = -1 # negative value indicates to use the xangle/beta* histogram

    if f:
      p.ctppsLHCInfo.xangleBetaStarHistogramFile = f
    if obj:
      p.ctppsLHCInfo.xangleBetaStarHistogramObject = obj
