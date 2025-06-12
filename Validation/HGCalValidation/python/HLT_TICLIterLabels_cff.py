import FWCore.ParameterSet.Config as cms

hltTiclIterLabels = ["hltTiclTrackstersCLUE3DHigh", "hltTiclTrackstersCLUE3DHighL1Seeded", "hltTiclTrackstersMerge"]

from Configuration.ProcessModifiers.ticl_v5_cff import ticl_v5
ticl_v5.toModify(
    globals(),
    lambda g: g.update({
        "hltTiclIterLabels": [
            "hltTiclTrackstersCLUE3DHigh",
            "hltTiclTrackstersCLUE3DHighL1Seeded",
            "hltTiclTracksterLinks",
            #"hltTiclTracksterLinksSuperclusteringDNNUnseeded",
            #"hltTiclTracksterLinksSuperclusteringDNNL1Seeded",
            "hltTiclCandidate"
        ]
    })
)

## remove the L1Seeded iteration form the HLT Ticl labels
from Configuration.ProcessModifiers.ngtScouting_cff import ngtScouting
_ngtLabels = [label for label in hltTiclIterLabels if label != "hltTiclTrackstersCLUE3DHighL1Seeded"]
ngtScouting.toModify(
    globals(), lambda g: g.update({"hltTiclIterLabels": _ngtLabels})
)
