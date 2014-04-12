import FWCore.ParameterSet.Config as cms

#RAW content 
SimG4CoreRAW = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_g4SimHits_*_*', 
        'keep edmHepMCProduct_source_*_*')
)
#RECO content
SimG4CoreRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep edmHepMCProduct_source_*_*', 
        'keep SimTracks_g4SimHits_*_*', 
        'keep SimVertexs_g4SimHits_*_*')
)
#AOD content
SimG4CoreAOD = cms.PSet(
    outputCommands = cms.untracked.vstring()
)

