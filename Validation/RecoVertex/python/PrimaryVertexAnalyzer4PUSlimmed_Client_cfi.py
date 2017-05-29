import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

def _resolPull(prefix):
    return [
        "{pfx}_ResolX_vs_PU '#sigma(x) vs PU' {pfx}_ResolX_vs_PU".format(pfx=prefix),
        "{pfx}_ResolY_vs_PU '#sigma(y) vs PU' {pfx}_ResolY_vs_PU".format(pfx=prefix),
        "{pfx}_ResolZ_vs_PU '#sigma(z) vs PU' {pfx}_ResolZ_vs_PU".format(pfx=prefix),
        "{pfx}_ResolPt2_vs_PU '#sigma(p_{{T}}^{{2}}) vs PU' {pfx}_ResolPt2_vs_PU".format(pfx=prefix),
        "{pfx}_PullX_vs_PU 'Pull x vs PU' {pfx}_PullX_vs_PU".format(pfx=prefix),
        "{pfx}_PullY_vs_PU 'Pull y vs PU' {pfx}_PullY_vs_PU".format(pfx=prefix),
        "{pfx}_PullZ_vs_PU 'Pull z vs PU' {pfx}_PullZ_vs_PU".format(pfx=prefix),

        "{pfx}_ResolX_vs_NumTracks '#sigma(x) vs NumTracks' {pfx}_ResolX_vs_NumTracks".format(pfx=prefix),
        "{pfx}_ResolY_vs_NumTracks '#sigma(y) vs NumTracks' {pfx}_ResolY_vs_NumTracks".format(pfx=prefix),
        "{pfx}_ResolZ_vs_NumTracks '#sigma(z) vs NumTracks' {pfx}_ResolZ_vs_NumTracks".format(pfx=prefix),
        "{pfx}_ResolPt2_vs_NumTracks '#sigma(p_{{T}}^{{2}}) vs NumTracks' {pfx}_ResolPt2_vs_NumTracks".format(pfx=prefix),
        "{pfx}_PullX_vs_NumTracks 'Pull x vs NumTracks' {pfx}_PullX_vs_NumTracks".format(pfx=prefix),
        "{pfx}_PullY_vs_NumTracks 'Pull y vs NumTracks' {pfx}_PullY_vs_NumTracks".format(pfx=prefix),
        "{pfx}_PullZ_vs_NumTracks 'Pull z vs NumTracks' {pfx}_PullZ_vs_NumTracks".format(pfx=prefix),

        "{pfx}_ResolX_vs_Z '#sigma(x) vs z' {pfx}_ResolX_vs_Z".format(pfx=prefix),
        "{pfx}_ResolY_vs_Z '#sigma(y) vs z' {pfx}_ResolY_vs_Z".format(pfx=prefix),
        "{pfx}_ResolZ_vs_Z '#sigma(z) vs z' {pfx}_ResolZ_vs_Z".format(pfx=prefix),
        "{pfx}_ResolPt2_vs_Z '#sigma(p_{{T}}^{{2}}) vs z' {pfx}_ResolPt2_vs_Z".format(pfx=prefix),
        "{pfx}_PullX_vs_Z 'Pull x vs z' {pfx}_PullX_vs_Z".format(pfx=prefix),
        "{pfx}_PullY_vs_Z 'Pull y vs z' {pfx}_PullY_vs_Z".format(pfx=prefix),
        "{pfx}_PullZ_vs_Z 'Pull z vs z' {pfx}_PullZ_vs_Z".format(pfx=prefix),

        "{pfx}_ResolX_vs_Pt '#sigma(x) vs p_{{T}}' {pfx}_ResolX_vs_Pt".format(pfx=prefix),
        "{pfx}_ResolY_vs_Pt '#sigma(y) vs p_{{T}}' {pfx}_ResolY_vs_Pt".format(pfx=prefix),
        "{pfx}_ResolZ_vs_Pt '#sigma(z) vs p_{{T}}' {pfx}_ResolZ_vs_Pt".format(pfx=prefix),
        "{pfx}_ResolPt2_vs_Pt '#sigma(p_{{T}}^{{2}}) vs p_{{T}}' {pfx}_ResolZ_vs_Pt".format(pfx=prefix),
        "{pfx}_PullX_vs_Pt 'Pull x vs p_{{T}}' {pfx}_PullX_vs_Pt".format(pfx=prefix),
        "{pfx}_PullY_vs_Pt 'Pull y vs p_{{T}}' {pfx}_PullY_vs_Pt".format(pfx=prefix),
        "{pfx}_PullZ_vs_Pt 'Pull z vs p_{{T}}' {pfx}_PullZ_vs_Pt".format(pfx=prefix),
    ]


postProcessorVertex = DQMEDHarvester("DQMGenericClient",
                                     subDirs = cms.untracked.vstring("Vertexing/PrimaryVertexV/*"),
                                     efficiency = cms.vstring(
                                         "effic_vs_NumVertices 'Efficiency vs NumVertices' GenAllAssoc2RecoMatched_NumVertices GenAllAssoc2Reco_NumVertices",
                                         "effic_vs_Z 'Efficiency vs Z' GenAllAssoc2RecoMatched_Z GenAllAssoc2Reco_Z",
                                         "effic_vs_R 'Efficiency vs R' GenAllAssoc2RecoMatched_R GenAllAssoc2Reco_R",
                                         "effic_vs_Pt2 'Efficiency vs Sum p_{T}^{2}' GenAllAssoc2RecoMatched_Pt2 GenAllAssoc2Reco_Pt2",
                                         "effic_vs_NumTracks 'Efficiency vs NumTracks' GenAllAssoc2RecoMatched_NumTracks GenAllAssoc2Reco_NumTracks",
                                         "effic_vs_ClosestVertexInZ 'Efficiency vs ClosestVtx in Z' GenAllAssoc2RecoMatched_ClosestDistanceZ GenAllAssoc2Reco_ClosestDistanceZ",
                                         "gen_duplicate_vs_ClosestVertexInZ 'Gen_Duplicate vs ClosestVtx in Z' GenAllAssoc2RecoMultiMatched_ClosestDistanceZ GenAllAssoc2Reco_ClosestDistanceZ",
                                         "gen_duplicate_vs_NumVertices 'Gen_Duplicate vs NumVertices' GenAllAssoc2RecoMultiMatched_NumVertices GenAllAssoc2Reco_NumVertices",
                                         "gen_duplicate_vs_Z 'Gen_Duplicate vs Z' GenAllAssoc2RecoMultiMatched_Z GenAllAssoc2Reco_Z",
                                         "gen_duplicate_vs_R 'Gen_Duplicate vs R' GenAllAssoc2RecoMultiMatched_R GenAllAssoc2Reco_R",
                                         "gen_duplicate_vs_Pt2 'Gen_Duplicate vs Sum p_{T}^{2}' GenAllAssoc2RecoMultiMatched_Pt2 GenAllAssoc2Reco_Pt2",
                                         "gen_duplicate_vs_NumTracks 'Gen_Duplicate vs NumTracks' GenAllAssoc2RecoMultiMatched_NumTracks GenAllAssoc2Reco_NumTracks",
                                         "gen_duplicate_vs_ClosestVertexInZ 'Gen_Duplicate vs ClosestVtx in Z' GenAllAssoc2RecoMultiMatched_ClosestDistanceZ GenAllAssoc2Reco_ClosestDistanceZ",
                                         "merged_vs_NumVertices 'Merged vs NumVertices' RecoAllAssoc2GenMultiMatched_NumVertices RecoAllAssoc2Gen_NumVertices",
                                         "merged_vs_PU 'Merged vs PU' RecoAllAssoc2GenMultiMatched_PU RecoAllAssoc2Gen_PU",
                                         "merged_vs_Z 'Merged vs Z' RecoAllAssoc2GenMultiMatched_Z RecoAllAssoc2Gen_Z",
                                         "merged_vs_R 'Merged vs R' RecoAllAssoc2GenMultiMatched_R RecoAllAssoc2Gen_R",
                                         "merged_vs_Pt2 'Merged vs Sum p_{T}^{2}' RecoAllAssoc2GenMultiMatched_Pt2 RecoAllAssoc2Gen_Pt2",
                                         "merged_vs_NumTracks 'Merged vs NumTracks' RecoAllAssoc2GenMultiMatched_NumTracks RecoAllAssoc2Gen_NumTracks",
                                         "merged_vs_ClosestVertexInZ 'Merged vs ClosestVtx in Z' RecoAllAssoc2GenMultiMatched_ClosestDistanceZ RecoAllAssoc2GenSimForMerge_ClosestDistanceZ",
                                         "fakerate_vs_NumVertices 'Fakerate vs NumVertices' RecoAllAssoc2GenMatched_NumVertices RecoAllAssoc2Gen_NumVertices fake",
                                         "fakerate_vs_PU 'Fakerate vs PU' RecoAllAssoc2GenMatched_PU RecoAllAssoc2Gen_PU fake",
                                         "fakerate_vs_Z 'Fakerate vs Z' RecoAllAssoc2GenMatched_Z RecoAllAssoc2Gen_Z fake",
                                         "fakerate_vs_R 'Fakerate vs R' RecoAllAssoc2GenMatched_R RecoAllAssoc2Gen_R fake",
                                         "fakerate_vs_Pt2 'Fakerate vs Sum p_{T}^{2}' RecoAllAssoc2GenMatched_Pt2 RecoAllAssoc2Gen_Pt2 fake",
                                         "fakerate_vs_Ndof 'Fakerate vs Ndof' RecoAllAssoc2GenMatched_Ndof RecoAllAssoc2Gen_Ndof fake",
                                         "fakerate_vs_NumTracks 'Fakerate vs NumTracks' RecoAllAssoc2GenMatched_NumTracks RecoAllAssoc2Gen_NumTracks fake",
                                         "fakerate_vs_ClosestVertexInZ 'Fakerate vs ClosestVtx in Z' RecoAllAssoc2GenMatched_ClosestDistanceZ RecoAllAssoc2Gen_ClosestDistanceZ fake",
                                         "fakerate_vs_Purity 'Fakerate vs Purity' RecoAllAssoc2GenMatched_Purity RecoAllAssoc2Gen_Purity fake",
                                         "duplicate_vs_NumVertices 'Duplicate vs NumVertices' RecoAllAssoc2MultiMatchedGen_NumVertices RecoAllAssoc2Gen_NumVertices",
                                         "duplicate_vs_PU 'Duplicate vs PU' RecoAllAssoc2MultiMatchedGen_PU RecoAllAssoc2Gen_PU",
                                         "duplicate_vs_Z 'Duplicate vs Z' RecoAllAssoc2MultiMatchedGen_Z RecoAllAssoc2Gen_Z",
                                         "duplicate_vs_R 'Duplicate vs R' RecoAllAssoc2MultiMatchedGen_R RecoAllAssoc2Gen_R",
                                         "duplicate_vs_Pt2 'Duplicate vs Sum p_{T}^{2}' RecoAllAssoc2MultiMatchedGen_Pt2 RecoAllAssoc2Gen_Pt2",
                                         "duplicate_vs_NumTracks 'Duplicate vs NumTracks' RecoAllAssoc2MultiMatchedGen_NumTracks RecoAllAssoc2Gen_NumTracks",
                                         "duplicate_vs_ClosestVertexInZ 'Duplicate vs ClosestsVtx In Z' RecoAllAssoc2MultiMatchedGen_ClosestDistanceZ RecoAllAssoc2Gen_ClosestDistanceZ",

                                         "PV_effic_vs_Z 'PV reco+tag efficiency vs Z' GenPVAssoc2RecoPVMatched_Z GenPVAssoc2RecoPV_Z",
                                     ),
                                     resolution = cms.vstring(
                                         _resolPull("RecoAllAssoc2GenMatched") +
                                         _resolPull("RecoAllAssoc2GenMatchedMerged") +
                                         _resolPull("RecoPVAssoc2GenPVMatched")
                                     ),
                                     outputFileName = cms.untracked.string(""),
                                     verbose = cms.untracked.uint32(5)
)
