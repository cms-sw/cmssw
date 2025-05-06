import FWCore.ParameterSet.Config as cms

#hltTiclIterLabels = ["hltTiclTrackstersCLUE3DHigh", "hltTiclTrackstersCLUE3DHighL1Seeded", "hltTiclTrackstersMerge"]
hltTiclIterLabels = ["hltTiclTrackstersCLUE3DHigh", "hltTiclTrackstersMerge"]

from Configuration.ProcessModifiers.ticl_v5_cff import ticl_v5
ticl_v5.toModify(
    globals(),
    lambda g: g.update({
        "hltTiclIterLabels": [
            "hltTiclTrackstersCLUE3DHigh",
            #"hltTiclTrackstersCLUE3DHighL1Seeded",
            "hltTiclTracksterLinks",
            #"hltTiclTracksterLinksSuperclusteringDNNUnseeded",
            #"hltTiclTracksterLinksSuperclusteringDNNL1Seeded",
            "hltTiclCandidate"
        ]
    })
)
