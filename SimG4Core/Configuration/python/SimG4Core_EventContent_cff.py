import FWCore.ParameterSet.Config as cms

#RAW content 
SimG4CoreRAW = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_g4SimHits_*_*', 
        'keep edmHepMCProduct_source_*_*',
        'keep recoGenParticles_RHDecayTracer_RHadronDecay_SIM')
)
#RECO content
SimG4CoreRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep edmHepMCProduct_source_*_*', 
        'keep SimTracks_g4SimHits_*_*', 
        'keep SimVertexs_g4SimHits_*_*',
        'keep recoGenParticles_RHDecayTracer_RHadronDecay_SIM')
)
#AOD content
SimG4CoreAOD = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoGenParticles_RHDecayTracer_RHadronDecay_SIM')
)
#HLTAODSIM content
SimG4CoreHLTAODSIM = cms.PSet(
    outputCommands = cms.untracked.vstring(
'keep SimVertexs_g4SimHits_*_*',
'keep recoGenParticles_RHDecayTracer_RHadronDecay_SIM')
)

