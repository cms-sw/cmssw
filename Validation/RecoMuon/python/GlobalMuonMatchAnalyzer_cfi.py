import FWCore.ParameterSet.Config as cms

# Put here the modules you want the cfg file to use,
# then include this file in your cfg file.
# i.e. in MatchValidator.cfg replace 'module demo = MatchValidator {} '
# with 'include "anlyzerDir/MatchValidator/data/MatchValidator.cfi" '.
# (Remember that filenames are case sensitive.)
globalMuonMatchAnalyzer = cms.EDAnalyzer("GlobalMuonMatchAnalyzer",
    muAssociator = cms.untracked.string('TrackAssociatorByPosition'),
    tkLabel = cms.untracked.InputTag("ctfWithMaterialTracks"),
    tkAssociator = cms.untracked.string('TrackAssociatorByHits'),
    tpLabel = cms.untracked.InputTag("cutsTPEffic"),
    muLabel = cms.untracked.InputTag("standAloneMuons","UpdatedAtVtx"),
    glbLabel = cms.untracked.InputTag("globalMuons"),
    out = cms.untracked.string('GlobalMatchValidator.root')
)



