import os
import collections
import six

_sampleName = {
    "RelValCloseByParticleGun_CE_H_Fine_300um" : "CloseByParticleGun in CE-H Fine section with 300 um",
    "RelValCloseByParticleGun_CE_H_Fine_200um" : "CloseByParticleGun in CE-H Fine section with 200 um",
    "RelValCloseByParticleGun_CE_H_Fine_120um" : "CloseByParticleGun in CE-H Fine section with 120 um",
    "RelValCloseByParticleGun_CE_H_Coarse_Scint" : "CloseByParticleGun in CE-H Coarse section with scintillator",
    "RelValCloseByParticleGun_CE_H_Coarse_300um" : "CloseByParticleGun in CE-H Coarse section with 300 um",
    "RelValCloseByParticleGun_CE_E_Front_300um" : "CloseByParticleGun in CE-E Front section with 300 um",
    "RelValCloseByParticleGun_CE_E_Front_200um" : "CloseByParticleGun in CE-E Front section with 200 um",
    "RelValCloseByPGun_CE_E_Front_120um" : "CloseByParticleGun in CE-E Front section with 120 um",
    "RelValCloseByPGun_CE_H_Fine_300um" : "CloseByParticleGun in CE-H Fine section with 300 um",
    "RelValCloseByPGun_CE_H_Fine_200um" : "CloseByParticleGun in CE-H Fine section with 200 um",
    "RelValCloseByPGun_CE_H_Fine_120um" : "CloseByParticleGun in CE-H Fine section with 120 um",
    "RelValCloseByPGun_CE_H_Coarse_Scint" : "CloseByParticleGun in CE-H Coarse section with scintillator",
    "RelValCloseByPGun_CE_H_Coarse_300um" : "CloseByParticleGun in CE-H Coarse section with 300 um",
    "RelValCloseByPGun_CE_E_Front_300um" : "CloseByParticleGun in CE-E Front section with 300 um",
    "RelValCloseByPGun_CE_E_Front_200um" : "CloseByParticleGun in CE-E Front section with 200 um",
    "RelValCloseByPGun_CE_E_Front_120um" : "CloseByParticleGun in CE-E Front section with 120 um",
    "RelValTTbar" : "TTbar",
    "RelValSingleGammaFlatPt8To150" : "Single Gamma Pt 8 GeV to 150 GeV ",
    "RelValSingleMuPt10" : "Single Muon Pt 10 GeV",
    "RelValSingleMuPt100" : "Single Muon Pt 100 GeV",
    "RelValSingleMuPt1000" : "Single Muon Pt 1000 GeV",
    "RelValSingleMuFlatPt2To100" : "Single Muon Pt 2 GeV to 100 GeV",
    "RelValSingleMuFlatPt0p7To10" : "Single Muon Pt 0.7 GeV to 10 GeV",
    "RelValSingleEFlatPt2To100" : "Single Electron Pt 2 GeV to 100 GeV",
    "RelValSingleTauFlatPt2To150" : "Single Tau Pt 2 GeV to 150 GeV",
    "RelValSinglePiFlatPt0p7To10" : "Single Pion Pt 0.7 GeV to 10 GeV",
    "RelValQCD_Pt20toInfMuEnrichPt15" : "QCD Pt 20 GeV to Inf with Muon Pt 15 GeV",
    "RelValQCD_Pt15To7000_Flat" : "QCD Pt 15 GeV to 7 TeV",
    "RelValZTT" : "ZTauTau",
    "RelValZMM" : "ZMuMu",
    "RelValZEE" : "ZEleEle",
    "RelValB0ToKstarMuMu" : "B0 To Kstar Muon Muon",
    "RelValBsToEleEle" : "Bs To Electron Electron",
    "RelValBsToMuMu" : "Bs To Muon Muon",
    "RelValBsToJpsiGamma" : "Bs To Jpsi Gamma",
    "RelValBsToJpsiPhi_mumuKK" : "Bs To JpsiPhi_mumuKK",
    "RelValBsToPhiPhi_KKKK" : "Bs To PhiPhi_KKKK",
    "RelValDisplacedMuPt30To100" : "Displaced Muon Pt 30 GeV to 100 GeV",
    "RelValDisplacedMuPt2To10" : "Displaced Muon Pt 2 GeV to 10 GeV",
    "RelValDisplacedMuPt10To30" : "Displaced Muon Pt 10 GeV to 30 GeV",
    "RelValTauToMuMuMu" : "Tau To Muon Muon Muon",
    "RelValMinBias" : "Min Bias",
    "RelValH125GGgluonfusion" : "Higgs to gamma gamma",
    "RelValNuGun" : "Neutrino gun",
    "RelValZpTT_1500" : "Z prime with 1500 GeV nominal mass",
    "RelValTenTau_15_500" : "Ten Taus with energy from 15 GeV to 500 GeV"
}

_sampleFileName = {
    "RelValCloseByParticleGun_CE_H_Fine_300um" : "closebycehf300",
    "RelValCloseByParticleGun_CE_H_Fine_200um" : "closebycehf200",
    "RelValCloseByParticleGun_CE_H_Fine_120um" : "closebycehf120",
    "RelValCloseByParticleGun_CE_H_Coarse_Scint" : "closebycehcscint",
    "RelValCloseByParticleGun_CE_H_Coarse_300um" : "closebycehc300",
    "RelValCloseByParticleGun_CE_E_Front_300um" : "closebyceef300",
    "RelValCloseByParticleGun_CE_E_Front_200um" : "closebyceef200",
    "RelValCloseByParticleGun_CE_E_Front_120um" : "closebyceef120",
    "RelValTTbar" : "ttbar",
    "RelValSingleGammaFlatPt8To150" : "gam8",
    "RelValSingleMuPt10" : "m10",
    "RelValSingleMuPt100" : "m100",
    "RelValSingleMuPt1000" : "m1000",
    "RelValSingleMuFlatPt2To100" : "mflat2t100",
    "RelValSingleMuFlatPt0p7To10" : "mflat0p7t10",
    "RelValSingleEFlatPt2To100" : "eflat2t100",
    "RelValSingleTauFlatPt2To150" : "tauflat2t150",
    "RelValSinglePiFlatPt0p7To10" : "piflat0p7t10",
    "RelValQCD_Pt20toInfMuEnrichPt15" : "qcd20enmu15",
    "RelValQCD_Pt15To7000_Flat" : "qcdflat15",
    "RelValZTT" : "ztautau",
    "RelValZMM" : "zmm",
    "RelValZEE" : "zee",
    "RelValB0ToKstarMuMu" : "b0kstmm",
    "RelValBsToEleEle" : "bsee",
    "RelValBsToMuMu" : "bsmm",
    "RelValBsToJpsiGamma" : "bsjpsg",
    "RelValBsToJpsiPhi_mumuKK" : "bsjpspmmkk",
    "RelValBsToPhiPhi_KKKK" : "bsjpsppkkkk",
    "RelValDisplacedMuPt30To100" : "dm30",
    "RelValDisplacedMuPt2To10" : "dm2",
    "RelValDisplacedMuPt10To30" : "dm10",
    "RelValTauToMuMuMu" : "taummm",
    "RelValMinBias" : "minbias",
    "RelValH125GGgluonfusion" : "hgg",
    "RelValNuGun" : "nug",
    "RelValZpTT_1500" : "zp1500tautau",
    "RelValTenTau_15_500" : "tentaus15to1500"

}


_pageNameMap = {
    "summary": "Summary",
    "hitCalibration": "Reconstructed hits calibration",
    "hitValidation" : "Simulated hits, digis, reconstructed hits validation" , 
    "hgcalLayerClusters": "Layer clusters",
    "ticlMultiClustersFromTrackstersEM": "Electromagnetic multiclusters",
    "ticlMultiClustersFromTrackstersHAD": "Hadronic multiclusters",
    "hgcalMultiClusters" : "Old multiclusters",
    "standalone" : "Standalone study on simulated hits, digis, reconstructed hits"   
}

_sectionNameMapOrder = collections.OrderedDict([
    # These are for the summary page
    # Will add later
    # hgcalLayerClusters
    ("hgcalLayerClusters", "Layer clusters"),
    # ticlMultiClustersFromTrackstersEM
    ("ticlMultiClustersFromTrackstersEM","Electromagnetic multiclusters"),
    # ticlMultiClustersFromTrackstersHAD
    ("ticlMultiClustersFromTrackstersHAD","Hadronic multiclusters"),
    # hgcalMultiClusters
    ("hgcalMultiClusters","Old multiclusters"),
])

#This is the summary section, where we define which plots will be shown in the summary page. 
_summary = {}

#Objects to keep in summary
_summobj = ['hitCalibration','hitValidation', 'hgcalLayerClusters','ticlMultiClustersFromTrackstersEM','ticlMultiClustersFromTrackstersHAD']
#_summobj = ['hitCalibration','hitValidation', 'hgcalLayerClusters']

#Plots to keep in summary from hitCalibration
summhitcalib=[
    'Layer_Occupancy/LayerOccupancy_LayerOccupancy.png',
    'ReconstructableEnergyOverCPenergy/ReconstructableEnergyOverCPenergy_h_EoP_CPene_300_calib_fraction.png',
    'ReconstructableEnergyOverCPenergy/ReconstructableEnergyOverCPenergy_h_EoP_CPene_200_calib_fraction.png',
    'ReconstructableEnergyOverCPenergy/ReconstructableEnergyOverCPenergy_h_EoP_CPene_100_calib_fraction.png',
    'ReconstructableEnergyOverCPenergy/ReconstructableEnergyOverCPenergy_h_EoP_CPene_scint_calib_fraction.png'
    ]

#Plots to keep in summary from hitValidation
summhitvalid = [
    'SimHits_Validation/HitValidation_heeEnSim.png',
    'SimHits_Validation/HitValidation_hebEnSim.png',
    'SimHits_Validation/HitValidation_hefEnSim.png']
                          
#Plots to keep in summary from layer clusters
summlc = [
    'Efficiencies_zminus/Efficiencies_zminus_globalEfficiencies.png' ,
    'Efficiencies_zplus/Efficiencies_zplus_globalEfficiencies.png' ,
    'Duplicates_zminus/Duplicates_zminus_globalEfficiencies.png' ,
    'Duplicates_zplus/Duplicates_zplus_globalEfficiencies.png' ,
    'FakeRate_zminus/FakeRate_zminus_globalEfficiencies.png' ,
    'FakeRate_zplus/FakeRate_zplus_globalEfficiencies.png' ,
    'MergeRate_zminus/MergeRate_zminus_globalEfficiencies.png' ,
    'MergeRate_zplus/MergeRate_zplus_globalEfficiencies.png',
    'SelectedCaloParticles_Photons/SelectedCaloParticles_num_caloparticle_eta.png',
    'SelectedCaloParticles_Photons/SelectedCaloParticles_caloparticle_pt.png',
    'SelectedCaloParticles_Photons/SelectedCaloParticles_caloparticle_phi.png',
    'SelectedCaloParticles_Photons/SelectedCaloParticles_caloparticle_energy.png',
    'SelectedCaloParticles_Photons/SelectedCaloParticles_Eta vs Zorigin.png'
    ]
                          

#Plots to keep in summary from ticlMultiClustersFromTrackstersEM
summmcEM = [
    'Efficiencies/Efficiencies_globalEfficiencies.png' ,
    'Duplicates/Duplicates_globalEfficiencies.png' ,
    'FakeRate/FakeRate_globalEfficiencies.png' ,
    'MergeRate/MergeRate_globalEfficiencies.png'
]

#Plots to keep in summary from ticlMultiClustersFromTrackstersHAD
summmcHAD = summmcEM

#Plots to keep in summary from standalone analysis
summstandalone = [
    'hgcalSimHitStudy/RZ_AllDetectors.png'                          
]

#Let's save the above for later
for obj in _summobj: 
    _summary[obj] = {}
_summary['hitCalibration'] = summhitcalib
_summary['hitValidation'] = summhitvalid
_summary['hgcalLayerClusters'] = summlc
_summary['ticlMultiClustersFromTrackstersEM'] = summmcEM
_summary['ticlMultiClustersFromTrackstersHAD'] = summmcHAD                          







