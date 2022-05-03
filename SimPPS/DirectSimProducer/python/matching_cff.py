import FWCore.ParameterSet.Config as cms

def matchDirectSimOutputs(process, AOD=False, miniAOD=False):
    process.totemRPUVPatternFinder.tagRecHit = cms.InputTag('ppsDirectProtonSimulation')
    process.ctppsPixelLocalTracks.tag = cms.InputTag('ppsDirectProtonSimulation')
    process.ctppsDiamondLocalTracks.recHitsTag = cms.InputTag('ppsDirectProtonSimulation')
    process.es_prefer_geometry = cms.ESPrefer('CTPPSCompositeESSource', 'ctppsCompositeESSource')
    if AOD:
        process.beamDivergenceVtxGenerator.src = cms.InputTag('')
        process.beamDivergenceVtxGenerator.srcGenParticle = cms.VInputTag(
            cms.InputTag('genPUProtons', 'genPUProtons'),
            cms.InputTag('genParticles')
        )
    elif miniAOD:
        process.beamDivergenceVtxGenerator.src = cms.InputTag('')
        process.beamDivergenceVtxGenerator.srcGenParticle = cms.VInputTag(
            cms.InputTag('genPUProtons', 'genPUProtons'),
            cms.InputTag('prunedGenParticles')
        )
