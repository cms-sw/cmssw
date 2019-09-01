import FWCore.ParameterSet.Config as cms

# supply beam parameters
from Validation.CTPPS.year_2016.ctppsBeamParametersESSource_cfi import *

# particle generator
from Validation.CTPPS.year_2016.randomXiThetaGunProducer_cfi import *

# direct simulation
from Validation.CTPPS.ctppsDirectProtonSimulation_cfi import *
ctppsDirectProtonSimulation.verbosity = 0
ctppsDirectProtonSimulation.hepMCTag = cms.InputTag('beamDivergenceVtxGenerator')
ctppsDirectProtonSimulation.useEmpiricalApertures = False
ctppsDirectProtonSimulation.roundToPitch = True
ctppsDirectProtonSimulation.pitchStrips = 66E-3 * 12 / 19 # effective value to reproduce real RP resolution
ctppsDirectProtonSimulation.produceHitsRelativeToBeam = True
ctppsDirectProtonSimulation.produceScoringPlaneHits = False
ctppsDirectProtonSimulation.produceRecHits = True

#----------------------------------------------------------------------------------------------------

def SetLevel1():
  ctppsBeamParametersESSource.vtxStddevX = 0E-4
  ctppsBeamParametersESSource.vtxStddevZ = 0

  ctppsBeamParametersESSource.beamDivX45 = 0E-6
  ctppsBeamParametersESSource.beamDivX56 = 0E-6
  ctppsBeamParametersESSource.beamDivY45 = 0E-6
  ctppsBeamParametersESSource.beamDivY56 = 0E-6

  ctppsDirectProtonSimulation.roundToPitch = False


def SetLevel2():
  ctppsBeamParametersESSource.beamDivX45 = 0E-6
  ctppsBeamParametersESSource.beamDivX56 = 0E-6
  ctppsBeamParametersESSource.beamDivY45 = 0E-6
  ctppsBeamParametersESSource.beamDivY56 = 0E-6

  ctppsDirectProtonSimulation.roundToPitch = False


def SetLevel3():
  ctppsDirectProtonSimulation.roundToPitch = False


def SetLevel4():
  pass

def SetLowTheta():
  process.generator.theta_x_sigma = 0E-6
  process.generator.theta_y_sigma = 0E-6


def SetLargeTheta():
  pass
