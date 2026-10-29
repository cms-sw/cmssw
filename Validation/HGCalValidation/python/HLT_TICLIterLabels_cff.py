import FWCore.ParameterSet.Config as cms


hltTiclIterLabelsPSet = cms.PSet(
    labels=cms.vstring(
        "hltTiclTrackstersCLUE3DHigh",
        "hltTiclTrackstersCLUE3DHighL1Seeded",
        "hltTiclTracksterLinks",
        "hltTiclTracksterLinksSuperclusteringDNNUnseeded",
        "hltTiclTracksterLinksSuperclusteringDNNL1Seeded",
        "hltTiclCandidate"
    )
)


# The NGT scouting menu has no L1-seeded TICL collections
from Configuration.ProcessModifiers.ngtScouting_cff import ngtScouting
ngtScouting.toModify(
    hltTiclIterLabelsPSet,
    labels=[label for label in hltTiclIterLabelsPSet.labels if "L1Seeded" not in label]
)
