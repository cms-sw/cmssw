import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

postProcessorRecoDisplacedMuon = DQMEDHarvester("DQMGenericClient",
    subDirs = cms.untracked.vstring("Muons/RecoDisplacedMuonV/RecoDisplacedMuon_MuonAssoc"),
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
postProcessorRecoDisplacedMuonGlb = postProcessorRecoDisplacedMuon.clone(
    subDirs = ["Muons/RecoDisplacedMuonV/RecoDisplacedMuon_MuonAssoc_Glb"]
)
postProcessorRecoDisplacedMuonTrk = postProcessorRecoDisplacedMuon.clone(
    subDirs = ["Muons/RecoDisplacedMuonV/RecoDisplacedMuon_MuonAssoc_Trk"]
)
postProcessorRecoDisplacedMuonSta = postProcessorRecoDisplacedMuon.clone(
    subDirs = ["Muons/RecoDisplacedMuonV/RecoDisplacedMuon_MuonAssoc_Sta"]
)
postProcessorRecoDisplacedMuonTgt = postProcessorRecoDisplacedMuon.clone(
    subDirs = ["Muons/RecoDisplacedMuonV/RecoDisplacedMuon_MuonAssoc_Tgt"]
)
postProcessorRecoDisplacedMuonGlbPF = postProcessorRecoDisplacedMuon.clone(
    subDirs = ["Muons/RecoDisplacedMuonV/RecoDisplacedMuon_MuonAssoc_GlbPF"]
)
postProcessorRecoDisplacedMuonTrkPF = postProcessorRecoDisplacedMuon.clone(
    subDirs = ["Muons/RecoDisplacedMuonV/RecoDisplacedMuon_MuonAssoc_TrkPF"]
)
postProcessorRecoDisplacedMuonStaPF = postProcessorRecoDisplacedMuon.clone(
    subDirs = ["Muons/RecoDisplacedMuonV/RecoDisplacedMuon_MuonAssoc_StaPF"]
)
#not sure about this one, which types are monitored
postProcessorRecoDisplacedMuonComp = DQMEDHarvester(
    "DQMGenericClient",
    subDirs = cms.untracked.vstring("Muons/RecoDisplacedMuonV/"),
    efficiency = cms.vstring(
    "Eff_GlbSta_Eta 'Eff_{GLB,STA} vs #eta' RecoDisplacedMuon_MuonAssoc_Glb/EffEta RecoDisplacedMuon_MuonAssoc_Sta/EffEta",
    "Eff_GlbSta_P   'Eff_{GLB,STA} vs p' RecoDisplacedMuon_MuonAssoc_Glb/EffP RecoDisplacedMuon_MuonAssoc_Sta/EffP",
    "Eff_GlbSta_Phi 'Eff_{GLB,STA} vs #phi' RecoDisplacedMuon_MuonAssoc_Glb/EffPhi RecoDisplacedMuon_MuonAssoc_Sta/EffPhi",
    "Eff_GlbSta_Pt  'Eff_{GLB,STA} vs p_{T}' RecoDisplacedMuon_MuonAssoc_Glb/EffPt RecoDisplacedMuon_MuonAssoc_Sta/EffPt",
    "Eff_TgtGlb_Eta 'Eff_{TGT,GLB} vs #eta' RecoDisplacedMuon_MuonAssoc_Tgt/EffEta RecoDisplacedMuon_MuonAssoc_Glb/EffEta",
    "Eff_TgtGlb_P   'Eff_{TGT,GLB} vs p' RecoDisplacedMuon_MuonAssoc_Tgt/EffP RecoDisplacedMuon_MuonAssoc_Glb/EffP",
    "Eff_TgtGlb_Phi 'Eff_{TGT,GLB} vs #phi' RecoDisplacedMuon_MuonAssoc_Tgt/EffPhi RecoDisplacedMuon_MuonAssoc_Glb/EffPhi",
    "Eff_TgtGlb_Pt  'Eff_{TGT,GLB} vs p_{T}' RecoDisplacedMuon_MuonAssoc_Tgt/EffPt RecoDisplacedMuon_MuonAssoc_Glb/EffPt",
    ),
    resolution = cms.vstring(""),
    outputFileName = cms.untracked.string("")
)

postProcessorRecoDisplacedMuonCompPF = DQMEDHarvester(
    "DQMGenericClient",
    subDirs = cms.untracked.vstring("Muons/RecoDisplacedMuonV/"),
    efficiency = cms.vstring(
    "Eff_GlbPFStaPF_Eta 'Eff_{GLBPF,STAPF} vs #eta' RecoDisplacedMuon_MuonAssoc_GlbPF/EffEta RecoDisplacedMuon_MuonAssoc_StaPF/EffEta",
    "Eff_GlbPFStaPF_P   'Eff_{GLBPF,STAPF} vs p' RecoDisplacedMuon_MuonAssoc_GlbPF/EffP RecoDisplacedMuon_MuonAssoc_StaPF/EffP",
    "Eff_GlbPFStaPF_Phi 'Eff_{GLBPF,STAPF} vs #phi' RecoDisplacedMuon_MuonAssoc_GlbPF/EffPhi RecoDisplacedMuon_MuonAssoc_StaPF/EffPhi",
    "Eff_GlbPFStaPF_Pt  'Eff_{GLBPF,STAPF} vs p_{T}' RecoDisplacedMuon_MuonAssoc_GlbPF/EffPt RecoDisplacedMuon_MuonAssoc_StaPF/EffPt",
    ),
    resolution = cms.vstring(""),
    outputFileName = cms.untracked.string("")
)

postProcessorsRecoDisplacedMuonValidator_seq = cms.Sequence( postProcessorRecoDisplacedMuonGlb 
                                                    * postProcessorRecoDisplacedMuonTrk 
                                                    * postProcessorRecoDisplacedMuonSta 
                                                    * postProcessorRecoDisplacedMuonTgt 
                                                    * postProcessorRecoDisplacedMuonGlbPF 
                                                    * postProcessorRecoDisplacedMuonTrkPF 
                                                    * postProcessorRecoDisplacedMuonStaPF 
                                                    * postProcessorRecoDisplacedMuonComp 
                                                    * postProcessorRecoDisplacedMuonCompPF )
