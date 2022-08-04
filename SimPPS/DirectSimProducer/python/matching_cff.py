import FWCore.ParameterSet.Config as cms

def matchDirectSimOutputs(process, AOD=False, miniAOD=False):
    # match sources of rechits with direct simulation outputs
    process.totemRPUVPatternFinder.tagRecHit = cms.InputTag('ppsDirectProtonSimulation')
    process.ctppsPixelLocalTracks.tag = cms.InputTag('ppsDirectProtonSimulation')
    process.ctppsDiamondLocalTracks.recHitsTag = cms.InputTag('ppsDirectProtonSimulation')
    # handle clashes between simulation and GT conditions
    process.es_prefer_composrc = cms.ESPrefer('CTPPSCompositeESSource', 'ctppsCompositeESSource')
    process.es_prefer_pixtopo = cms.ESPrefer('PPSPixelTopologyESSource', 'ppsPixelTopologyESSource')
    process.es_prefer_lhcinfo = cms.ESPrefer('CTPPSBeamParametersFromLHCInfoESSource', 'ctppsBeamParametersFromLHCInfoESSource')
    process.es_prefer_assocuts = cms.ESPrefer('PPSAssociationCutsESSource', 'ppsAssociationCutsESSource')
    return process

def matchDirectSimOutputsAOD(process):
    matchDirectSimOutputs(process)
    process.beamDivergenceVtxGenerator.src = cms.InputTag('')
    process.beamDivergenceVtxGenerator.srcGenParticle = cms.VInputTag(
        cms.InputTag('genPUProtons', 'genPUProtons'),
        cms.InputTag('genParticles')
    )
    return process

def matchDirectSimOutputsMiniAOD(process):
    matchDirectSimOutputs(process)
    process.beamDivergenceVtxGenerator.src = cms.InputTag('')
    process.beamDivergenceVtxGenerator.srcGenParticle = cms.VInputTag(
        cms.InputTag('genPUProtons', 'genPUProtons'),
        cms.InputTag('prunedGenParticles')
    )
    return process
