import FWCore.ParameterSet.Config as cms
from CalibPPS.ESProducers.ctppsBeamParametersFromLHCInfoESSource_cfi import ctppsBeamParametersFromLHCInfoESSource as _esLHCinfo
from SimPPS.DirectSimProducer.ppsDirectProtonSimulation_cfi import ppsDirectProtonSimulation as _dirProtonSim
from IOMC.EventVertexGenerators.beamDivergenceVtxGenerator_cfi import beamDivergenceVtxGenerator as _vtxGen

# vertex smearing
beamDivergenceVtxGenerator = _vtxGen.clone()

# beam parameters as determined by PPS
ctppsBeamParametersFromLHCInfoESSource = _esLHCinfo.clone(
    lhcInfoLabel = "",
    # beam divergence (rad)
    beamDivX45 = 30.e-6,
    beamDivX56 = 30.e-6,
    beamDivY45 = 30.e-6,
    beamDivY56 = 30.e-6,
    # vertex offset (cm)
    vtxOffsetX45 = 0.,
    vtxOffsetX56 = 0.,
    vtxOffsetY45 = 0.,
    vtxOffsetY56 = 0.,
    vtxOffsetZ45 = 0.,
    vtxOffsetZ56 = 0.,
    # vertex sigma (cm)
    vtxStddevX = 1.e-3,
    vtxStddevY = 1.e-3,
    vtxStddevZ = 5.
)

# direct simulation
ppsDirectProtonSimulation = _dirProtonSim.clone(
    hepMCTag = 'beamDivergenceVtxGenerator',
    pitchStrips = 66.e-3 * 12 / 19,  # effective value to reproduce real RP resolution
    pitchPixelsHor = 50.e-3,
    pitchPixelsVer = 80.e-3,
    produceScoringPlaneHits = False,
)
