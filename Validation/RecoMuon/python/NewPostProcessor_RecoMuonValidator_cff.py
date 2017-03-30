import FWCore.ParameterSet.Config as cms

NEWpostProcessorRecoMuon = cms.EDAnalyzer("DQMGenericClient",
    subDirs = cms.untracked.vstring("Muons/RecoMuonV/NewRecoMuon_MuonAssoc"),
    #efficiencies and fractions
    efficiency = cms.vstring("EffP   'Efficiency vs p'     P   SimP  ",
                             "EffPt  'Efficiency vs p_{T}' Pt  SimPt ",
                             "EffEta 'Efficiency vs #eta'  Eta SimEta",
                             "EffPhi 'Efficiency vs #phi'  Phi SimPhi",
                             "MisQProbPt  'Charge Mis-identification probability vs p_{T}' MisQPt  SimPt ",
                             "MisQProbEta 'Charge Mis-identification probability vs #eta'  MisQEta SimEta",
                             #fractions
                             "FractP   'Muontype fraction vs p'     PMuon   PMuonAll",
                             "FractPt  'Muontype fraction  vs p_{T}' PtMuon PtMuonAll",
                             "FractEta 'Muontype fraction vs #eta'  EtaMuon EtaMuonAll",
                             "FractPhi 'Muontype fraction vs #phi'  PhiMuon PhiMuonAll",
                             ),

    resolution = cms.vstring("ErrP_vs_P      '#sigma(p) vs p'           ErrP_vs_P     ",
                             "ErrP_vs_Eta    '#sigma(p) vs #eta'        ErrP_vs_Eta   ",
                             "ErrPt_vs_Pt    '#sigma(p_{T}) vs p_{T}'   ErrPt_vs_Pt   ",
                             "ErrPt_vs_Eta   '#sigma(p_{T}) vs #eta'    ErrPt_vs_Eta  ",
                             "ErrEta_vs_Eta  '#sigma(#eta) vs #eta '    ErrEta_vs_Eta ",
                             "ErrQPt_vs_Pt   '#sigma(q/p_{T}) vs p_{T}' ErrQPt_vs_Pt  ",
                             "ErrQPt_vs_Eta  '#sigma(q/p_{T}) vs #eta'  ErrQPt_vs_Eta ",
                             "PullEta_vs_Pt  'Pull of #eta vs p_{T}'    PullEta_vs_Pt ",
                             "PullEta_vs_Eta 'Pull of #eta vs #eta'     PullEta_vs_Eta",
                             "PullPhi_vs_Eta 'Pull of #phi vs #eta'     PullPhi_vs_Eta",
                             "PullPt_vs_Pt   'Pull of p_{T} vs p_{T}'   PullPt_vs_Pt  ",
                             "PullPt_vs_Eta  'Pull of p_{T} vs #eta'    PullPt_vs_Eta ",
                             ),    
    outputFileName = cms.untracked.string("")
)

# for each type monitored
NEWpostProcessorRecoMuon_Glb = NEWpostProcessorRecoMuon.clone()
NEWpostProcessorRecoMuon_Glb.subDirs = cms.untracked.vstring("Muons/RecoMuonV/NewRecoMuon_MuonAssoc_Glb")

NEWpostProcessorRecoMuon_Trk = NEWpostProcessorRecoMuon.clone()
NEWpostProcessorRecoMuon_Trk.subDirs = cms.untracked.vstring("Muons/RecoMuonV/NewRecoMuon_MuonAssoc_Trk")

NEWpostProcessorRecoMuon_Sta = NEWpostProcessorRecoMuon.clone()
NEWpostProcessorRecoMuon_Sta.subDirs = cms.untracked.vstring("Muons/RecoMuonV/NewRecoMuon_MuonAssoc_Sta")

NEWpostProcessorRecoMuon_Tgt = NEWpostProcessorRecoMuon.clone()
NEWpostProcessorRecoMuon_Tgt.subDirs = cms.untracked.vstring("Muons/RecoMuonV/NewRecoMuon_MuonAssoc_Tgt")

NEWpostProcessorRecoMuon_GlbPF = NEWpostProcessorRecoMuon.clone()
NEWpostProcessorRecoMuon_GlbPF.subDirs = cms.untracked.vstring("Muons/RecoMuonV/NewRecoMuon_MuonAssoc_GlbPF")

NEWpostProcessorRecoMuon_TrkPF = NEWpostProcessorRecoMuon.clone()
NEWpostProcessorRecoMuon_TrkPF.subDirs = cms.untracked.vstring("Muons/RecoMuonV/NewRecoMuon_MuonAssoc_TrkPF")

NEWpostProcessorRecoMuon_StaPF = NEWpostProcessorRecoMuon.clone()
NEWpostProcessorRecoMuon_StaPF.subDirs = cms.untracked.vstring("Muons/RecoMuonV/NewRecoMuon_MuonAssoc_StaPF")

#not sure about this one, which types are monitored
NEWpostProcessorRecoMuonComp = cms.EDAnalyzer(
    "DQMGenericClient",
    subDirs = cms.untracked.vstring("Muons/RecoMuonV/"),
    efficiency = cms.vstring(
    "NewEff_GlbSta_Eta 'Eff_{GLB,STA} vs #eta' NewRecoMuon_MuonAssoc_Glb/EffEta NewRecoMuon_MuonAssoc_Sta/EffEta",
    "NewEff_GlbSta_P   'Eff_{GLB,STA} vs p' NewRecoMuon_MuonAssoc_Glb/EffP NewRecoMuon_MuonAssoc_Sta/EffP",
    "NewEff_GlbSta_Phi 'Eff_{GLB,STA} vs #phi' NewRecoMuon_MuonAssoc_Glb/EffPhi NewRecoMuon_MuonAssoc_Sta/EffPhi",
    "NewEff_GlbSta_Pt  'Eff_{GLB,STA} vs p_{T}' NewRecoMuon_MuonAssoc_Glb/EffPt NewRecoMuon_MuonAssoc_Sta/EffPt",
    "NewEff_TgtGlb_Eta 'Eff_{TGT,GLB} vs #eta' NewRecoMuon_MuonAssoc_Tgt/EffEta NewRecoMuon_MuonAssoc_Glb/EffEta",
    "NewEff_TgtGlb_P   'Eff_{TGT,GLB} vs p' NewRecoMuon_MuonAssoc_Tgt/EffP NewRecoMuon_MuonAssoc_Glb/EffP",
    "NewEff_TgtGlb_Phi 'Eff_{TGT,GLB} vs #phi' NewRecoMuon_MuonAssoc_Tgt/EffPhi NewRecoMuon_MuonAssoc_Glb/EffPhi",
    "NewEff_TgtGlb_Pt  'Eff_{TGT,GLB} vs p_{T}' NewRecoMuon_MuonAssoc_Tgt/EffPt NewRecoMuon_MuonAssoc_Glb/EffPt",
    ),
    resolution = cms.vstring(""),
    outputFileName = cms.untracked.string("")
)

NEWpostProcessorRecoMuonCompPF = cms.EDAnalyzer(
    "DQMGenericClient",
    subDirs = cms.untracked.vstring("Muons/RecoMuonV/"),
    efficiency = cms.vstring(
    "NewEff_GlbPFStaPF_Eta 'Eff_{GLBPF,STAPF} vs #eta' NewRecoMuon_MuonAssoc_GlbPF/EffEta NewRecoMuon_MuonAssoc_StaPF/EffEta",
    "NewEff_GlbPFStaPF_P   'Eff_{GLBPF,STAPF} vs p' NewRecoMuon_MuonAssoc_GlbPF/EffP NewRecoMuon_MuonAssoc_StaPF/EffP",
    "NewEff_GlbPFStaPF_Phi 'Eff_{GLBPF,STAPF} vs #phi' NewRecoMuon_MuonAssoc_GlbPF/EffPhi NewRecoMuon_MuonAssoc_StaPF/EffPhi",
    "NewEff_GlbPFStaPF_Pt  'Eff_{GLBPF,STAPF} vs p_{T}' NewRecoMuon_MuonAssoc_GlbPF/EffPt NewRecoMuon_MuonAssoc_StaPF/EffPt",
    ),
    resolution = cms.vstring(""),
    outputFileName = cms.untracked.string("")
)

NEWpostProcessorsRecoMuonValidator_seq = cms.Sequence( NEWpostProcessorRecoMuon_Glb 
                                                    * NEWpostProcessorRecoMuon_Trk 
                                                    * NEWpostProcessorRecoMuon_Sta 
                                                    * NEWpostProcessorRecoMuon_Tgt 
                                                    * NEWpostProcessorRecoMuon_GlbPF 
                                                    * NEWpostProcessorRecoMuon_TrkPF 
                                                    * NEWpostProcessorRecoMuon_StaPF 
                                                    * NEWpostProcessorRecoMuonComp 
                                                    * NEWpostProcessorRecoMuonCompPF )
