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

  process.ppsDirectProtonSimulation.roundToPitch = False

def SetSmearingLevel2(obj):
  obj.beamDivX45 = 0E-6
  obj.beamDivX56 = 0E-6
  obj.beamDivY45 = 0E-6
  obj.beamDivY56 = 0E-6

def SetLevel2(process):
  SetSmearingLevel2(process.ctppsBeamParametersFromLHCInfoESSource)

  process.ppsDirectProtonSimulation.roundToPitch = False

def SetLevel3(process):
  process.ppsDirectProtonSimulation.roundToPitch = False

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

