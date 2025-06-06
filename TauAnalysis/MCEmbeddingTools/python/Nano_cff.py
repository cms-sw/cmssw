import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.common_cff import ExtVar
from PhysicsTools.NanoAOD.l1trig_cff import *  # l1TablesTask, and all producers saved in the task need to be imported
from PhysicsTools.NanoAOD.nano_cff import *  # nanoSequence, nanoTableTaskFS, also all tasks and producerst saved in the sequence need to be imported

full_sim_nanoAOD_seq = cms.Sequence(nanoTableTaskFS)
l1_nanoAOD_seq = cms.Sequence(l1TablesTask)


""" Write some information about the, for the tau embedding method, selected muons to the NanoAOD."""
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
embeddingTableTask = cms.Task(embeddingTable)
embeddingTable_seq = cms.Sequence(embeddingTableTask)
embedding_nanoAOD_seq = cms.Sequence(
    l1_nanoAOD_seq
    + full_sim_nanoAOD_seq
    + nanoSequence
    + embeddingTable_seq
)