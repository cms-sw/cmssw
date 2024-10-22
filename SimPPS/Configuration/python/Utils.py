import FWCore.ParameterSet.Config as cms

def setupPPSDirectSim(process):
    process.load('SimPPS.Configuration.directSimPPS_cff')
    process.load('RecoPPS.Configuration.recoCTPPS_cff')
    if not hasattr(process, 'RandomNumberGeneratorService'):
        process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService")
    if not hasattr(process.RandomNumberGeneratorService, 'beamDivergenceVtxGenerator'):
        process.RandomNumberGeneratorService.beamDivergenceVtxGenerator = cms.PSet(initialSeed = cms.untracked.uint32(3849))
    if not hasattr(process.RandomNumberGeneratorService, 'ppsDirectProtonSimulation'):
        process.RandomNumberGeneratorService.ppsDirectProtonSimulation = cms.PSet(initialSeed = cms.untracked.uint32(4981))
    process.ppsDirectSim = cms.Path(process.directSimPPS * process.recoDirectSimPPS)
    process.schedule.append(process.ppsDirectSim)

    from SimPPS.DirectSimProducer.matching_cff import matchDirectSimOutputs
    matchDirectSimOutputs(process)
    return process

def setupPPSDirectSimAOD(process):
    setupPPSDirectSim(process)
    from SimPPS.DirectSimProducer.matching_cff import matchDirectSimOutputsAOD
    matchDirectSimOutputsAOD(process)
    return process

def setupPPSDirectSimMiniAOD(process):
    setupPPSDirectSim(process)
    from SimPPS.DirectSimProducer.matching_cff import matchDirectSimOutputsMiniAOD
    matchDirectSimOutputsMiniAOD(process)
    return process
