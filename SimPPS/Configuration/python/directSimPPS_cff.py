import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras
from CalibPPS.ESProducers.ctppsBeamParametersFromLHCInfoESSource_cfi import ctppsBeamParametersFromLHCInfoESSource as _esLHCinfo
from SimPPS.DirectSimProducer.ppsDirectProtonSimulation_cfi import ppsDirectProtonSimulation as _dirProtonSim

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
    hepMCTag = cms.InputTag('beamDivergenceVtxGenerator'),
    pitchStrips = 66.e-3 * 12 / 19,  # effective value to reproduce real RP resolution
    pitchPixelsHor = 50.e-3,
    pitchPixelsVer = 80.e-3,
    produceScoringPlaneHits = False,
)

ppsDirectSimTask = cms.Task(
    ppsDirectProtonSimulation,
)

ppsDirectSim = cms.Sequence(ppsDirectSimTask)

# modify according to era

def _modify2016(process):
    print('Process customised for 2016 PPS era')
    process.load('SimPPS.DirectSimProducer.simPPS2016_cfi')

def _modify2017(process):
    print('Process customised for 2017 PPS era')
    process.load('SimPPS.DirectSimProducer.simPPS2017_cfi')

def _modify2018(process):
    print('Process customised for 2018 PPS era')
    process.load('SimPPS.DirectSimProducer.simPPS2018_cfi')

def _modify2021(process):
    print('Process customised for 2021 PPS era')
    process.load('SimPPS.DirectSimProducer.simPPS2021_cfi')

modifyConfigurationStandardSequencesFor2016_ = eras.ctpps_2016.makeProcessModifier(_modify2016)
modifyConfigurationStandardSequencesFor2017_ = eras.ctpps_2017.makeProcessModifier(_modify2017)
modifyConfigurationStandardSequencesFor2018_ = eras.ctpps_2018.makeProcessModifier(_modify2018)
modifyConfigurationStandardSequencesFor2021_ = eras.ctpps_2021.makeProcessModifier(_modify2021)
