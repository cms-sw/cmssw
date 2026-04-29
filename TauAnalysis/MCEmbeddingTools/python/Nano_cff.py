"""
This config fragment is used to generate NanoAOD samples from tau embedding MiniAOD samples.
It adds an additional table with information about the initially selected muons and the initial event.

As the tau embedding event is a hybrid event and we want to have information about the measured event and the generated tau decay,
Both, the normal NanoAOD sequence for measured events and the NanoAOD sequence for simulated events, have to run.
This and the embeddingTable_seq is configured in the autoNANO dict in PhysicsTools/NanoAOD/python/autoNANO.py
With this one can simply add `NANO:@TauEmbedding` to the `--step` parameter of the cmsDriver command, to get all the additional tables.
The merging step must be carried out beforehand.
To use this config fragment, a cmsDriver command like the following can be used:
```
cmsDriver.py \
	--step NANO:@TauEmbedding \
	--data \
	--scenario pp \
	--eventcontent TauEmbeddingNANOAOD \
	--datatier NANOAODSIM \
	--era ... \
    --conditions ... \
    --filein ... \
    --fileout ...
```
"""

import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.common_cff import ExtVar

# This table producer adds information about the initially selected muons and the initial event to the NanoAOD.
# The information comes from the `selectedMuonsForEmbedding` producer, which is created in the selection step.
embeddingTable = cms.EDProducer(
    "GlobalVariablesTableProducer",
    name=cms.string("TauEmbedding"),
    # doc=cms.string("TauEmbedding"),
    variables=cms.PSet(
        ptLeadingMuon=ExtVar(
            cms.InputTag("selectedMuonsForEmbedding", "ptLeadingMuon"),
            float,
            doc="leading muon pt",
        ),
        ptTrailingMuon=ExtVar(
            cms.InputTag("selectedMuonsForEmbedding", "ptTrailingMuon"),
            float,
            doc="trailing muon pt",
        ),
        etaLeadingMuon=ExtVar(
            cms.InputTag("selectedMuonsForEmbedding", "etaLeadingMuon"),
            float,
            doc="leading muon eta",
        ),
        etaTrailingMuon=ExtVar(
            cms.InputTag("selectedMuonsForEmbedding", "etaTrailingMuon"),
            float,
            doc="trailing muon eta",
        ),
        phiLeadingMuon=ExtVar(
            cms.InputTag("selectedMuonsForEmbedding", "phiLeadingMuon"),
            float,
            doc="leading muon phi",
        ),
        phiTrailingMuon=ExtVar(
            cms.InputTag("selectedMuonsForEmbedding", "phiTrailingMuon"),
            float,
            doc="trailing muon phi",
        ),
        chargeLeadingMuon=ExtVar(
            cms.InputTag("selectedMuonsForEmbedding", "chargeLeadingMuon"),
            float,
            doc="leading muon charge",
        ),
        chargeTrailingMuon=ExtVar(
            cms.InputTag("selectedMuonsForEmbedding", "chargeTrailingMuon"),
            float,
            doc="trailing muon charge",
        ),
        dbLeadingMuon=ExtVar(
            cms.InputTag("selectedMuonsForEmbedding", "dbLeadingMuon"),
            float,
            doc="leading muon DB",
        ),
        dbTrailingMuon=ExtVar(
            cms.InputTag("selectedMuonsForEmbedding", "dbTrailingMuon"),
            float,
            doc="trailing muon DB",
        ),
        massLeadingMuon=ExtVar(
            cms.InputTag("selectedMuonsForEmbedding", "massLeadingMuon"),
            float,
            doc="leading muon mass",
        ),
        massTrailingMuon=ExtVar(
            cms.InputTag("selectedMuonsForEmbedding", "massTrailingMuon"),
            float,
            doc="trailing muon mass",
        ),
        vtxXLeadingMuon=ExtVar(
            cms.InputTag("selectedMuonsForEmbedding", "vtxxLeadingMuon"),
            float,
            doc="leading muon vertex X",
        ),
        vtxYLeadingMuon=ExtVar(
            cms.InputTag("selectedMuonsForEmbedding", "vtxyLeadingMuon"),
            float,
            doc="leading muon vertex Y",
        ),
        vtxZLeadingMuon=ExtVar(
            cms.InputTag("selectedMuonsForEmbedding", "vtxzLeadingMuon"),
            float,
            doc="leading muon vertex Z",
        ),
        vtxXTrailingMuon=ExtVar(
            cms.InputTag("selectedMuonsForEmbedding", "vtxxTrailingMuon"),
            float,
            doc="trailing muon vertex X",
        ),
        vtxYTrailingMuon=ExtVar(
            cms.InputTag("selectedMuonsForEmbedding", "vtxyTrailingMuon"),
            float,
            doc="trailing muon vertex Y",
        ),
        vtxZTrailingMuon=ExtVar(
            cms.InputTag("selectedMuonsForEmbedding", "vtxzTrailingMuon"),
            float,
            doc="trailing muon vertex Z",
        ),
        isMediumLeadingMuon=ExtVar(
            cms.InputTag("selectedMuonsForEmbedding", "isMediumLeadingMuon"),
            bool,
            doc="leading muon ID (medium)",
        ),
        isMediumTrailingMuon=ExtVar(
            cms.InputTag("selectedMuonsForEmbedding", "isMediumTrailingMuon"),
            bool,
            doc="trailing muon ID (medium)",
        ),
        isTightLeadingMuon=ExtVar(
            cms.InputTag("selectedMuonsForEmbedding", "isTightLeadingMuon"),
            bool,
            doc="leading muon ID (tight)",
        ),
        isTightTrailingMuon=ExtVar(
            cms.InputTag("selectedMuonsForEmbedding", "isTightTrailingMuon"),
            bool,
            doc="trailing muon ID (tight)",
        ),
        initialMETEt=ExtVar(
            cms.InputTag("selectedMuonsForEmbedding", "initialMETEt"),
            float,
            doc="MET Et of selected event",
        ),
        initialMETphi=ExtVar(
            cms.InputTag("selectedMuonsForEmbedding", "initialMETphi"),
            float,
            doc="MET phi of selected event",
        ),
        initialPuppiMETEt=ExtVar(
            cms.InputTag("selectedMuonsForEmbedding", "initialPuppiMETEt"),
            float,
            doc="PuppiMET Et of selected event",
        ),
        initialPuppiMETphi=ExtVar(
            cms.InputTag("selectedMuonsForEmbedding", "initialPuppiMETphi"),
            float,
            doc="PuppiMET phi of selected event",
        ),
    ),
)
# put all sequences and tasks together to create the embedding NanoAOD sequence
embeddingTableTask = cms.Task(embeddingTable)
embeddingTable_seq = cms.Sequence(embeddingTableTask)
