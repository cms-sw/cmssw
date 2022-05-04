import FWCore.ParameterSet.Config as cms

def setupPPSDirectSimAOD(process):
    process.load('SimPPS.Configuration.directSimPPS_cff')
    process.load('RecoPPS.Configuration.recoCTPPS_cff')
    from SimPPS.DirectSimProducer.matching_cff import matchDirectSimOutputsAOD
    matchDirectSimOutputsAOD(process)
    process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
        beamDivergenceVtxGenerator = cms.PSet(initialSeed = cms.untracked.uint32(3849)),
        ppsDirectProtonSimulation = cms.PSet(initialSeed = cms.untracked.uint32(4981))
    )
    process.directSim = cms.Path(process.directSimPPS * process.recoDirectSimPPS)
    process.schedule.append(process.directSim)
    return process

def setupPPSDirectSimMiniAOD(process):
    process.load('SimPPS.Configuration.directSimPPS_cff')
    process.load('RecoPPS.Configuration.recoCTPPS_cff')
    from SimPPS.DirectSimProducer.matching_cff import matchDirectSimOutputsMiniAOD
    matchDirectSimOutputsMiniAOD(process)
    process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
        beamDivergenceVtxGenerator = cms.PSet(initialSeed = cms.untracked.uint32(3849)),
        ppsDirectProtonSimulation = cms.PSet(initialSeed = cms.untracked.uint32(4981))
    )
    process.directSim = cms.Path(process.directSimPPS * process.recoDirectSimPPS)
    process.schedule.append(process.directSim)
    return process
