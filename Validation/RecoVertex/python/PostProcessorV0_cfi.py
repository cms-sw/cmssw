#   "effic 'Efficiency vs #eta' num_assoc(simToReco)_eta num_simul_eta",

import FWCore.ParameterSet.Config as cms

postProcessorV0 = cms.EDAnalyzer("DQMGenericClient",
    subDirs = cms.untracked.vstring("Vertexing/V0V/EffFakes/*"),
    efficiency = cms.vstring(
       "K0sEffVsR 'K^{0}_{S} efficiency vs R (radial)' K0sEffVsR_num K0sEffVsR_denom", 
       "K0sEffVsEta 'K^{0}_{S} efficiency vs #eta' K0sEffVsEta_num K0sEffVsEta_denom",
       "K0sEffVsPt 'K^{0}_{S} efficiency vs p_{T}' K0sEffVsPt_num K0sEffVsPt_denom",
       "LamEffVsR '#Lambda efficiency vs R (radial)' LamEffVsR_num LamEffVsR_denom",
       "LamEffVsEta '#Lambda efficiency vs #eta' LamEffVsEta_num LamEffVsEta_denom",
       "LamEffVsPt '#Lambda efficiency vs p_{T}' LamEffVsPt_num LamEffVsPt_denom",
       "K0sFakeVsR 'K^{0}_{S} fake rate vs R (radial)' K0sFakeVsR_num K0sFakeVsR_denom",
       "K0sFakeVsEta 'K^{0}_{S} fake rate vs #eta' K0sFakeVsEta_num K0sFakeVsEta_denom",
       "K0sFakeVsPt 'K^{0}_{S} fake rate vs p_{T}' K0sFakeVsPt_num K0sFakeVsPt_denom",
       "LamFakeVsR '#Lambda fake rate vs R (radial)' LamFakeVsR_num LamFakeVsR_denom",
       "LamFakeVsEta '#Lambda fake rate vs #eta' LamFakeVsEta_num LamFakeVsEta_denom",
       "LamFakeVsPt '#Lambda fake rate vs p{T}' LamFakeVsPt_num LamFakeVsPt_denom",
       "K0sTkEffVsR 'K^{0}_{S} tracking efficiency vs R (radial)' K0sTkEffVsR_num K0sEffVsR_denom", 
       "K0sTkEffVsEta 'K^{0}_{S} tracking efficiency vs #eta' K0sTkEffVsEta_num K0sEffVsEta_denom",
       "K0sTkEffVsPt 'K^{0}_{S} tracking efficiency vs p_{T}' K0sTkEffVsPt_num K0sEffVsPt_denom",
       "LamTkEffVsR '#Lambda tracking efficiency vs R (radial)' LamTkEffVsR_num LamEffVsR_denom",
       "LamTkEffVsEta '#Lambda tracking efficiency vs #eta' LamTkEffVsEta_num LamEffVsEta_denom",
       "LamTkEffVsPt '#Lambda tracking efficiency vs p_{T}' LamTkEffVsPt_num LamEffVsPt_denom",
       "K0sTkFakeVsR 'K^{0}_{S} tracking fake rate vs R (radial)' K0sTkFakeVsR_num K0sFakeVsR_denom",
       "K0sTkFakeVsEta 'K^{0}_{S} tracking fake rate vs #eta' K0sTkFakeVsEta_num K0sFakeVsEta_denom",
       "K0sTkFakeVsPt 'K^{0}_{S} tracking fake rate vs p_{T}' K0sTkFakeVsPt_num K0sFakeVsPt_denom",
       "LamTkFakeVsR '#Lambda tracking fake rate vs R (radial)' LamTkFakeVsR_num LamFakeVsR_denom",
       "LamTkFakeVsEta '#Lambda tracking fake rate vs #eta' LamTkFakeVsEta_num LamFakeVsEta_denom",
       "LamTkFakeVsPt '#Lambda tracking fake rate vs p{T}' LamTkFakeVsPt_num LamFakeVsPt_denom"
   ),
   resolution = cms.vstring(),
   outputFileName = cms.untracked.string("")
)
