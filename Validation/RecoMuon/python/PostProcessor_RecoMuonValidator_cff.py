import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

postProcessorRecoMuon = DQMEDHarvester("DQMGenericClient",
    subDirs = cms.untracked.vstring("Muons/RecoMuonV/RecoMuon_MuonAssoc"),
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
postProcessorRecoMuonGlb = postProcessorRecoMuon.clone()
postProcessorRecoMuonGlb.subDirs = cms.untracked.vstring("Muons/RecoMuonV/RecoMuon_MuonAssoc_Glb")

postProcessorRecoMuonTrk = postProcessorRecoMuon.clone()
postProcessorRecoMuonTrk.subDirs = cms.untracked.vstring("Muons/RecoMuonV/RecoMuon_MuonAssoc_Trk")

postProcessorRecoMuonSta = postProcessorRecoMuon.clone()
postProcessorRecoMuonSta.subDirs = cms.untracked.vstring("Muons/RecoMuonV/RecoMuon_MuonAssoc_Sta")

postProcessorRecoMuonTgt = postProcessorRecoMuon.clone()
postProcessorRecoMuonTgt.subDirs = cms.untracked.vstring("Muons/RecoMuonV/RecoMuon_MuonAssoc_Tgt")

postProcessorRecoMuonGlbPF = postProcessorRecoMuon.clone()
postProcessorRecoMuonGlbPF.subDirs = cms.untracked.vstring("Muons/RecoMuonV/RecoMuon_MuonAssoc_GlbPF")

postProcessorRecoMuonTrkPF = postProcessorRecoMuon.clone()
postProcessorRecoMuonTrkPF.subDirs = cms.untracked.vstring("Muons/RecoMuonV/RecoMuon_MuonAssoc_TrkPF")

postProcessorRecoMuonStaPF = postProcessorRecoMuon.clone()
postProcessorRecoMuonStaPF.subDirs = cms.untracked.vstring("Muons/RecoMuonV/RecoMuon_MuonAssoc_StaPF")

#not sure about this one, which types are monitored
postProcessorRecoMuonComp = DQMEDHarvester(
    "DQMGenericClient",
    subDirs = cms.untracked.vstring("Muons/RecoMuonV/"),
    efficiency = cms.vstring(
    "Eff_GlbSta_Eta 'Eff_{GLB,STA} vs #eta' RecoMuon_MuonAssoc_Glb/EffEta RecoMuon_MuonAssoc_Sta/EffEta",
    "Eff_GlbSta_P   'Eff_{GLB,STA} vs p' RecoMuon_MuonAssoc_Glb/EffP RecoMuon_MuonAssoc_Sta/EffP",
    "Eff_GlbSta_Phi 'Eff_{GLB,STA} vs #phi' RecoMuon_MuonAssoc_Glb/EffPhi RecoMuon_MuonAssoc_Sta/EffPhi",
    "Eff_GlbSta_Pt  'Eff_{GLB,STA} vs p_{T}' RecoMuon_MuonAssoc_Glb/EffPt RecoMuon_MuonAssoc_Sta/EffPt",
    "Eff_TgtGlb_Eta 'Eff_{TGT,GLB} vs #eta' RecoMuon_MuonAssoc_Tgt/EffEta RecoMuon_MuonAssoc_Glb/EffEta",
    "Eff_TgtGlb_P   'Eff_{TGT,GLB} vs p' RecoMuon_MuonAssoc_Tgt/EffP RecoMuon_MuonAssoc_Glb/EffP",
    "Eff_TgtGlb_Phi 'Eff_{TGT,GLB} vs #phi' RecoMuon_MuonAssoc_Tgt/EffPhi RecoMuon_MuonAssoc_Glb/EffPhi",
    "Eff_TgtGlb_Pt  'Eff_{TGT,GLB} vs p_{T}' RecoMuon_MuonAssoc_Tgt/EffPt RecoMuon_MuonAssoc_Glb/EffPt",
    ),
    resolution = cms.vstring(""),
    outputFileName = cms.untracked.string("")
)

postProcessorRecoMuonCompPF = DQMEDHarvester(
    "DQMGenericClient",
    subDirs = cms.untracked.vstring("Muons/RecoMuonV/"),
    efficiency = cms.vstring(
    "Eff_GlbPFStaPF_Eta 'Eff_{GLBPF,STAPF} vs #eta' RecoMuon_MuonAssoc_GlbPF/EffEta RecoMuon_MuonAssoc_StaPF/EffEta",
    "Eff_GlbPFStaPF_P   'Eff_{GLBPF,STAPF} vs p' RecoMuon_MuonAssoc_GlbPF/EffP RecoMuon_MuonAssoc_StaPF/EffP",
    "Eff_GlbPFStaPF_Phi 'Eff_{GLBPF,STAPF} vs #phi' RecoMuon_MuonAssoc_GlbPF/EffPhi RecoMuon_MuonAssoc_StaPF/EffPhi",
    "Eff_GlbPFStaPF_Pt  'Eff_{GLBPF,STAPF} vs p_{T}' RecoMuon_MuonAssoc_GlbPF/EffPt RecoMuon_MuonAssoc_StaPF/EffPt",
    ),
    resolution = cms.vstring(""),
    outputFileName = cms.untracked.string("")
)

postProcessorsRecoMuonValidator_seq = cms.Sequence( postProcessorRecoMuonGlb 
                                                    * postProcessorRecoMuonTrk 
                                                    * postProcessorRecoMuonSta 
                                                    * postProcessorRecoMuonTgt 
                                                    * postProcessorRecoMuonGlbPF 
                                                    * postProcessorRecoMuonTrkPF 
                                                    * postProcessorRecoMuonStaPF 
                                                    * postProcessorRecoMuonComp 
                                                    * postProcessorRecoMuonCompPF )
