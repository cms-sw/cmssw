import FWCore.ParameterSet.Config as cms

postProcessorMuonMultiTrack = cms.EDAnalyzer("DQMGenericClient",
    subDirs = cms.untracked.vstring("Muons/RecoMuonV/MultiTrack/*"),
    efficiency = cms.vstring(
    "effic 'Efficiency vs #eta' num_assoc(simToReco)_eta num_simul_eta",
    "efficPt 'Efficiency vs p_{T}' num_assoc(simToReco)_pT num_simul_pT",
    "effic_vs_hit 'Efficiency vs n Hits' num_assoc(simToReco)_hit num_simul_hit",
    ),
    resolutionLimitedFit = cms.untracked.bool(False),
    resolution = cms.vstring("cotThetares_vs_eta '#sigma(cot(#theta)) vs #eta' cotThetares_vs_eta",
                             "cotThetares_vs_pt '#sigma(cot(#theta)) vs p_{T}' cotThetares_vs_pt",
                             "dxypull_vs_eta 'd_{xy} Pull vs #eta' dxypull_vs_eta",
                             "dxyres_vs_eta '#sigma(d_{xy}) vs #eta' dxyres_vs_eta",
                             "dxyres_vs_pt '#sigma(d_{xy}) vs p_{T}' dxyres_vs_pt",
                             "dzpull_vs_eta 'd_{z} Pull vs #eta' dzpull_vs_eta",
                             "dzres_vs_eta '#sigma(d_{z}) vs #eta' dzres_vs_eta",
                             "dzres_vs_pt '#sigma(d_{z}) vs p_{T}' dzres_vs_pt",
                             "etares_vs_eta '#sigma(#eta) vs #eta' etares_vs_eta",
                             "phipull_vs_eta '#phi Pull vs #eta' phipull_vs_eta",
                             "phipull_vs_phi '#phi Pull vs #phi' phipull_vs_phi",
                             "phires_vs_eta '#sigma(#phi) vs #eta' phires_vs_eta",
                             "phires_vs_phi '#sigma(#phi) vs #phi' phires_vs_phi",
                             "phires_vs_pt '#sigma(#phi) vs p_{T}' phires_vs_pt",
                             "ptpull_vs_eta 'p_{T} Pull vs #eta' ptpull_vs_eta",
                             "ptpull_vs_phi 'p_{T} Pull vs #phi' ptpull_vs_phi",
                             "ptres_vs_eta '#sigma(p_{T}) vs #eta' ptres_vs_eta",
                             "ptres_vs_phi '#sigma(p_{T}) vs #phi' ptres_vs_phi",
                             "ptres_vs_pt '#sigma(p_{T}) vs p_{T}' ptres_vs_pt",
                             "thetapull_vs_eta '#theta Pull vs #eta' thetapull_vs_eta",
                             "thetapull_vs_phi '#theta Pull vs #phi' thetapull_vs_phi"),
    outputFileName = cms.untracked.string("")
)

postProcessorMuonMultiTrackComp = cms.EDAnalyzer("DQMGenericClient",
    subDirs = cms.untracked.vstring("Muons/RecoMuonV/MultiTrack/"),
    efficiency = cms.vstring(
    "Eff_GlbTk_Eta 'Eff_{GLB,TK} vs #eta' globalMuons_tpToGlbAssociation/effic general_tpToTkmuAssociation/effic",
    "Eff_GlbTk_Pt 'Eff_{GLB,TK} vs p_{T}' globalMuons_tpToGlbAssociation/efficPt general_tpToTkmuAssociation/efficPt",
    "Eff_GlbTk_Hit 'Eff_{GLB,TK} vs n Hits' globalMuons_tpToGlbAssociation/effic_vs_hit general_tpToTkmuAssociation/effic_vs_hit",
    "Eff_GlbSta_Eta 'Eff_{GLB,STA} vs #eta' globalMuons_tpToGlbAssociation/effic standAloneMuons_UpdatedAtVtx_tpToStaUpdAssociation/effic",
    "Eff_GlbSta_Pt 'Eff_{GLB,STA} vs p_{T}' globalMuons_tpToGlbAssociation/efficPt standAloneMuons_UpdatedAtVtx_tpToStaUpdAssociation/efficPt",
    "Eff_GlbSta_Hit 'Eff_{GLB,STA} vs n Hits' globalMuons_tpToGlbAssociation/effic_vs_hit standAloneMuons_UpdatedAtVtx_tpToStaUpdAssociation/effic_vs_hit",

    "Eff_GlbTk_Eta_mabh 'Eff_{GLB,TK} vs #eta' globalMuons_tpToGlbMuonAssociation/effic general_tpToTkMuonAssociation/effic",
    "Eff_GlbTk_Pt_mabh 'Eff_{GLB,TK} vs p_{T}' globalMuons_tpToGlbMuonAssociation/efficPt general_tpToTkMuonAssociation/efficPt",
    "Eff_GlbTk_Hit_mabh 'Eff_{GLB,TK} vs n Hits' globalMuons_tpToGlbMuonAssociation/effic_vs_hit general_tpToTkMuonAssociation/effic_vs_hit",
    "Eff_GlbSta_Eta_mabh 'Eff_{GLB,STA} vs #eta' globalMuons_tpToGlbMuonAssociation/effic standAloneMuons_UpdatedAtVtx_tpToStaUpdMuonAssociation/effic",
    "Eff_GlbSta_Pt_mabh 'Eff_{GLB,STA} vs p_{T}' globalMuons_tpToGlbMuonAssociation/efficPt standAloneMuons_UpdatedAtVtx_tpToStaUpdMuonAssociation/efficPt",
    "Eff_GlbSta_Hit_mabh 'Eff_{GLB,STA} vs n Hits' globalMuons_tpToGlbMuonAssociation/effic_vs_hit standAloneMuons_UpdatedAtVtx_tpToStaUpdMuonAssociation/effic_vs_hit",
    ),
    resolution = cms.vstring(""),
    outputFileName = cms.untracked.string("")
)

postProcessorMuonMultiTrackCompFS = cms.EDAnalyzer("DQMGenericClient",
    subDirs = cms.untracked.vstring("Muons/RecoMuonV/MultiTrack/"),
    efficiency = cms.vstring(
    "Eff_GlbTk_Eta 'Eff_{GLB,TK} vs #eta' globalMuons_tpToGlbAssociationFS/effic general_tpToTkmuAssociationFS/effic",
    "Eff_GlbTk_Pt 'Eff_{GLB,TK} vs p_{T}' globalMuons_tpToGlbAssociationFS/efficPt general_tpToTkmuAssociationFS/efficPt",
    "Eff_GlbTk_Hit 'Eff_{GLB,TK} vs n Hits' globalMuons_tpToGlbAssociationFS/effic_vs_hit general_tpToTkmuAssociationFS/effic_vs_hit",
    "Eff_GlbSta_Eta 'Eff_{GLB,STA} vs #eta' globalMuons_tpToGlbAssociationFS/effic standAloneMuons_UpdatedAtVtx_tpToStaUpdAssociationFS/effic",
    "Eff_GlbSta_Pt 'Eff_{GLB,STA} vs p_{T}' globalMuons_tpToGlbAssociationFS/efficPt standAloneMuons_UpdatedAtVtx_tpToStaUpdAssociationFS/efficPt",
    "Eff_GlbSta_Hit 'Eff_{GLB,STA} vs n Hits' globalMuons_tpToGlbAssociationFS/effic_vs_hit standAloneMuons_UpdatedAtVtx_tpToStaUpdAssociationFS/effic_vs_hit",

    "Eff_GlbTk_Eta_mabh 'Eff_{GLB,TK} vs #eta' globalMuons_tpToGlbMuonAssociationFS/effic general_tpToTkMuonAssociationFS/effic",
    "Eff_GlbTk_Pt_mabh 'Eff_{GLB,TK} vs p_{T}' globalMuons_tpToGlbMuonAssociationFS/efficPt general_tpToTkMuonAssociationFS/efficPt",
    "Eff_GlbTk_Hit_mabh 'Eff_{GLB,TK} vs n Hits' globalMuons_tpToGlbMuonAssociationFS/effic_vs_hit general_tpToTkMuonAssociationFS/effic_vs_hit",
    "Eff_GlbSta_Eta_mabh 'Eff_{GLB,STA} vs #eta' globalMuons_tpToGlbMuonAssociationFS/effic standAloneMuons_UpdatedAtVtx_tpToStaUpdMuonAssociationFS/effic",
    "Eff_GlbSta_Pt_mabh 'Eff_{GLB,STA} vs p_{T}' globalMuons_tpToGlbMuonAssociationFS/efficPt standAloneMuons_UpdatedAtVtx_tpToStaUpdMuonAssociationFS/efficPt",
    "Eff_GlbSta_Hit_mabh 'Eff_{GLB,STA} vs n Hits' globalMuons_tpToGlbMuonAssociationFS/effic_vs_hit standAloneMuons_UpdatedAtVtx_tpToStaUpdMuonAssociationFS/effic_vs_hit",
    ),
    resolution = cms.vstring(""),
    outputFileName = cms.untracked.string("")
)


postProcessorRecoMuon = cms.EDAnalyzer("DQMGenericClient",
    subDirs = cms.untracked.vstring("Muons/RecoMuonV/RecoMuon_*",),

    efficiency = cms.vstring("Trk/EffP   'Efficiency vs p'     Trk/P   Muons/SimP  ",
                             "Trk/EffPt  'Efficiency vs p_{T}' Trk/Pt  Muons/SimPt ",
                             "Trk/EffEta 'Efficiency vs #eta'  Trk/Eta Muons/SimEta",
                             "Trk/EffPhi 'Efficiency vs #phi'  Trk/Phi Muons/SimPhi",
                             "Trk/MisQProbPt  'Charge Mis-identification probability vs p_{T}' Trk/MisQPt  Muons/SimPt ",
                             "Trk/MisQProbEta 'Charge Mis-identification probability vs #eta'  Trk/MisQEta Muons/SimEta",
                             "Sta/EffP   'Efficiency vs p'     Sta/P   Muons/SimP  ",
                             "Sta/EffPt  'Efficiency vs p_{T}' Sta/Pt  Muons/SimPt ",
                             "Sta/EffEta 'Efficiency vs #eta'  Sta/Eta Muons/SimEta",
                             "Sta/EffPhi 'Efficiency vs #phi'  Sta/Phi Muons/SimPhi",
                             "Sta/MisQProbPt  'Charge Mis-identification probability vs p_{T}' Sta/MisQPt  Muons/SimPt ",
                             "Sta/MisQProbEta 'Charge Mis-identification probability vs #eta'  Sta/MisQEta Muons/SimEta",

                             "Glb/EffP   'Efficiency vs p'     Glb/P   Muons/SimP  ",
                             "Glb/EffPt  'Efficiency vs p_{T}' Glb/Pt  Muons/SimPt ",
                             "Glb/EffEta 'Efficiency vs #eta'  Glb/Eta Muons/SimEta",
                             "Glb/EffPhi 'Efficiency vs #phi'  Glb/Phi Muons/SimPhi",
                             "Glb/MisQProbPt  'Charge Mis-identification probability vs p_{T}' Glb/MisQPt  Muons/SimPt ",
                             "Glb/MisQProbEta 'Charge Mis-identification probability vs #eta'  Glb/MisQEta Muons/SimEta",
                             ),

    resolution = cms.vstring("Trk/ErrP_vs_P      '#sigma(p) vs p'           Trk/ErrP_vs_P     ",
                             "Trk/ErrP_vs_Eta    '#sigma(p) vs #eta'        Trk/ErrP_vs_Eta   ",
                             "Trk/ErrPt_vs_Pt    '#sigma(p) vs p_{T}'       Trk/ErrPt_vs_Pt   ",
                             "Trk/ErrPt_vs_Eta   '#sigma(p) vs #eta'        Trk/ErrPt_vs_Eta  ",
                             "Trk/ErrEta_vs_Eta  '#sigma(#eta) vs #eta '    Trk/ErrEta_vs_Eta ",
                             "Trk/ErrQPt_vs_Pt   '#sigma(q/p_{T}) vs p_{T}' Trk/ErrQPt_vs_Pt  ",
                             "Trk/ErrQPt_vs_Eta  '#sigma(q/p_{T}) vs #eta'  Trk/ErrQPt_vs_Eta ",
                             "Trk/PullEta_vs_Pt  'Pull of #eta vs p_{T}'    Trk/PullEta_vs_Pt ",
                             "Trk/PullEta_vs_Eta 'Pull of #eta vs #eta'     Trk/PullEta_vs_Eta",
                             "Trk/PullPhi_vs_Eta 'Pull of #phi vs #eta'     Trk/PullPhi_vs_Eta",
                             "Trk/PullPt_vs_Pt   'Pull of p_{T} vs p_{T}'   Trk/PullPt_vs_Pt  ",
                             "Trk/PullPt_vs_Eta  'Pull of p_{T} vs #eta'    Trk/PullPt_vs_Eta ",

                             "Sta/ErrP_vs_P      '#sigma(p) vs p'           Sta/ErrP_vs_P     ",
                             "Sta/ErrP_vs_Eta    '#sigma(p) vs #eta'        Sta/ErrP_vs_Eta   ",
                             "Sta/ErrPt_vs_Pt    '#sigma(p) vs p_{T}'       Sta/ErrPt_vs_Pt   ",
                             "Sta/ErrPt_vs_Eta   '#sigma(p) vs #eta'        Sta/ErrPt_vs_Eta  ",
                             "Sta/ErrEta_vs_Eta  '#sigma(#eta) vs #eta '    Sta/ErrEta_vs_Eta ",
                             "Sta/ErrQPt_vs_Pt   '#sigma(q/p_{T}) vs p_{T}' Sta/ErrQPt_vs_Pt  ",
                             "Sta/ErrQPt_vs_Eta  '#sigma(q/p_{T}) vs #eta'  Sta/ErrQPt_vs_Eta ",
                             "Sta/PullEta_vs_Pt  'Pull of #eta vs p_{T}'    Sta/PullEta_vs_Pt ",
                             "Sta/PullEta_vs_Eta 'Pull of #eta vs #eta'     Sta/PullEta_vs_Eta",
                             "Sta/PullPhi_vs_Eta 'Pull of #phi vs #eta'     Sta/PullPhi_vs_Eta",
                             "Sta/PullPt_vs_Pt   'Pull of p_{T} vs p_{T}'   Sta/PullPt_vs_Pt  ",
                             "Sta/PullPt_vs_Eta  'Pull of p_{T} vs #eta'    Sta/PullPt_vs_Eta ",

                             "Glb/ErrP_vs_P      '#sigma(p) vs p'           Glb/ErrP_vs_P     ",
                             "Glb/ErrP_vs_Eta    '#sigma(p) vs #eta'        Glb/ErrP_vs_Eta   ",
                             "Glb/ErrPt_vs_Pt    '#sigma(p) vs p_{T}'       Glb/ErrPt_vs_Pt   ",
                             "Glb/ErrPt_vs_Eta   '#sigma(p) vs #eta'        Glb/ErrPt_vs_Eta  ",
                             "Glb/ErrEta_vs_Eta  '#sigma(#eta) vs #eta '    Glb/ErrEta_vs_Eta ",
                             "Glb/ErrQPt_vs_Pt   '#sigma(q/p_{T}) vs p_{T}' Glb/ErrQPt_vs_Pt  ",
                             "Glb/ErrQPt_vs_Eta  '#sigma(q/p_{T}) vs #eta'  Glb/ErrQPt_vs_Eta ",
                             "Glb/PullEta_vs_Pt  'Pull of #eta vs p_{T}'    Glb/PullEta_vs_Pt ",
                             "Glb/PullEta_vs_Eta 'Pull of #eta vs #eta'     Glb/PullEta_vs_Eta",
                             "Glb/PullPhi_vs_Eta 'Pull of #phi vs #eta'     Glb/PullPhi_vs_Eta",
                             "Glb/PullPt_vs_Pt   'Pull of p_{T} vs p_{T}'   Glb/PullPt_vs_Pt  ",
                             "Glb/PullPt_vs_Eta  'Pull of p_{T} vs #eta'    Glb/PullPt_vs_Eta ",
                             ),
    outputFileName = cms.untracked.string("")
)

postProcessorRecoMuonComp = cms.EDAnalyzer(
    "DQMGenericClient",
    subDirs = cms.untracked.vstring("Muons/RecoMuonV/RecoMuon_MuonAssoc", "Muons/RecoMuonV/RecoMuon_TrackAssoc",),
    efficiency = cms.vstring(
    "Eff_GlbTrk_Eta 'Eff_{GLB,TK} vs #eta' Glb/EffEta Trk/EffEta",
    "Eff_GlbTrk_P 'Eff_{GLB,TK} vs p' Glb/EffP Trk/EffP",
    "Eff_GlbTrk_Phi 'Eff_{GLB,TK} vs #phi' Glb/EffPhi Trk/EffPhi",
    "Eff_GlbTrk_Pt 'Eff_{GLB,TK} vs p_{T}' Glb/EffPt Trk/EffPt",
    
    "Eff_GlbSta_Eta 'Eff_{GLB,TK} vs #eta' Glb/EffEta Sta/EffEta",
    "Eff_GlbSta_P 'Eff_{GLB,TK} vs p' Glb/EffP Sta/EffP",
    "Eff_GlbSta_Phi 'Eff_{GLB,TK} vs #phi' Glb/EffPhi Sta/EffPhi",
    "Eff_GlbSta_Pt 'Eff_{GLB,TK} vs p_{T}' Glb/EffPt Sta/EffPt",
    ),
    resolution = cms.vstring(""),
    outputFileName = cms.untracked.string("")
)

        

recoMuonPostProcessors = cms.Sequence(postProcessorMuonMultiTrack*postProcessorRecoMuon*postProcessorMuonMultiTrackComp*postProcessorRecoMuonComp)

recoMuonPostProcessorsFastSim = cms.Sequence(postProcessorMuonMultiTrack*postProcessorRecoMuon*postProcessorMuonMultiTrackCompFS*postProcessorRecoMuonComp)
