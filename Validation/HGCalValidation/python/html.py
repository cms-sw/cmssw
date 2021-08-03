import os
import collections

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
    "layerClusters": "Layer clusters",
    "tracksters":"Tracksters", 
    "ticlMultiClustersFromTrackstersEM": "Electromagnetic multiclusters",
    "ticlMultiClustersFromTrackstersHAD": "Hadronic multiclusters",
    "hgcalMultiClusters" : "Old multiclusters",
    "standalone" : "Standalone study on simulated hits, digis, reconstructed hits"   
}

_sectionNameMapOrder = collections.OrderedDict([
    # These are for the summary page
    # Will add later
    # layerClusters
    ("layerClusters", "Layer clusters"),
    # ticlMultiClustersFromTrackstersEM
    ("ticlMultiClustersFromTrackstersEM","Electromagnetic multiclusters"),
    # ticlMultiClustersFromTrackstersHAD
    ("ticlMultiClustersFromTrackstersHAD","Hadronic multiclusters"),
    ("tracksters","Tracksters"),
    # hgcalMultiClusters
    ("hgcalMultiClusters","Old multiclusters"),
])

#This is the summary section, where we define which plots will be shown in the summary page. 
_summary = {}

#Objects to keep in summary
#_summobj = ['hitCalibration','hitValidation', 'hgcalLayerClusters','ticlMultiClustersFromTrackstersEM','ticlMultiClustersFromTrackstersHAD']
_summobj = ['hitCalibration','hitValidation', 'layerClusters','tracksters']
#_summobj = ['hitCalibration','hitValidation', 'layerClusters']

#Plots to keep in summary from hitCalibration
summhitcalib=[
    'Layer_Occupancy/LayerOccupancy/LayerOccupancy.png',
    'ReconstructableEnergyOverCPenergy/ReconstructableEnergyOverCPenergy/h_EoP_CPene_300_calib_fraction.png',
    'ReconstructableEnergyOverCPenergy/ReconstructableEnergyOverCPenergy/h_EoP_CPene_200_calib_fraction.png',
    'ReconstructableEnergyOverCPenergy/ReconstructableEnergyOverCPenergy/h_EoP_CPene_100_calib_fraction.png',
    'ReconstructableEnergyOverCPenergy/ReconstructableEnergyOverCPenergy/h_EoP_CPene_scint_calib_fraction.png'
    ]

#Plots to keep in summary from hitValidation
summhitvalid = [
    'SimHits_Validation/HitValidation/heeEnSim.png',
    'SimHits_Validation/HitValidation/hebEnSim.png',
    'SimHits_Validation/HitValidation/hefEnSim.png']
                          
#Plots to keep in summary from layer clusters
summlc = [
    'hgcalLayerClusters_Z-minus: LC_CP association/Efficiencies_vs_layer/globalEfficiencies.png' ,
    'hgcalLayerClusters_Z-plus: LC_CP association/Efficiencies_vs_layer/globalEfficiencies.png' ,
    'hgcalLayerClusters_Z-minus: LC_CP association/Duplicates_vs_layer/globalEfficiencies.png' ,
    'hgcalLayerClusters_Z-plus: LC_CP association/Duplicates_vs_layer/globalEfficiencies.png' ,
    'hgcalLayerClusters_Z-minus: LC_CP association/FakeRate_vs_layer/globalEfficiencies.png' ,
    'hgcalLayerClusters_Z-plus: LC_CP association/FakeRate_vs_layer/globalEfficiencies.png' ,
    'hgcalLayerClusters_Z-minus: LC_CP association/MergeRate_vs_layer/globalEfficiencies.png' ,
    'hgcalLayerClusters_Z-plus: LC_CP association/MergeRate_vs_layer/globalEfficiencies.png',
    'SelectedCaloParticles_Photons/SelectedCaloParticles_num_caloparticle_eta.png',
    'SelectedCaloParticles_Photons/SelectedCaloParticles_caloparticle_pt.png',
    'SelectedCaloParticles_Photons/SelectedCaloParticles_caloparticle_phi.png',
    'SelectedCaloParticles_Photons/SelectedCaloParticles_caloparticle_energy.png',
    'SelectedCaloParticles_Photons/SelectedCaloParticles_Eta vs Zorigin.png'
    ]

#Plots to keep in summary from ticlMultiClustersFromTrackstersEM
summmcEM = [
    'ticlTrackstersEM/Purities/globalEfficiencies.png' ,
    'ticlTrackstersEM/Duplicates/globalEfficiencies.png' ,
    'ticlTrackstersEM/FakeRate/globalEfficiencies.png' ,
    'ticlTrackstersEM/MergeRate/globalEfficiencies.png'
]

#Plots to keep in summary from ticlMultiClustersFromTrackstersHAD
summmcHAD = [
    'ticlTrackstersHAD/Purities/globalEfficiencies.png' ,
    'ticlTrackstersHAD/Duplicates/globalEfficiencies.png' ,
    'ticlTrackstersHAD/FakeRate/globalEfficiencies.png' ,
    'ticlTrackstersHAD/MergeRate/globalEfficiencies.png'
]

summmcTICL = summmcEM + summmcHAD

#Plots to keep in summary from standalone analysis
summstandalone = [
    'hgcalSimHitStudy/RZ_AllDetectors.png'                          
]

#Let's save the above for later
for obj in _summobj: 
    _summary[obj] = {}
_summary['hitCalibration'] = summhitcalib
_summary['hitValidation'] = summhitvalid
_summary['layerClusters'] = summlc
_summary['tracksters'] = summmcTICL
#_summary['allTiclMultiClusters'] = summmcTICL
#_summary['ticlMultiClustersFromTrackstersEM'] = summmcEM
#_summary['ticlMultiClustersFromTrackstersHAD'] = summmcHAD                          

#Entering the geometry section 
#_MatBudSections = ["allhgcal","zminus","zplus","indimat","fromvertex"]
_MatBudSections = ["allhgcal","indimat","fromvertex"]

_geoPageNameMap = {
 "allhgcal": "All materials",
# "zminus" : "Zminus",
# "zplus"  : "Zplus",
 "indimat" : "Individual materials",
 "fromvertex": "From vertex up to in front of muon stations"    
}

_individualmaterials =['Air','Aluminium','Cables','Copper','Epoxy','HGC_G10-FR4','Kapton','Lead','Other','Scintillator','Silicon','Stainless_Steel','WCu']

_matPageNameMap = {
 'Air': 'Air',
 'Aluminium': 'Aluminium',
 'Cables': 'Cables',
 'Copper': 'Copper',
 'Epoxy': 'Epoxy',
 'HGC_G10-FR4': 'HGC_G10-FR4',
 'Kapton': 'Kapton',
 'Lead': 'Lead',
 'Other': 'Other',
 'Scintillator': 'Scintillator',
 'Silicon': 'Silicon',
 'Stainless_Steel': 'Stainless Steel',
 'WCu': 'WCu'
}

_individualmatplots = {"HGCal_x_vs_z_vs_Rsum","HGCal_l_vs_z_vs_Rsum","HGCal_x_vs_z_vs_Rsumcos","HGCal_l_vs_z_vs_Rsumcos","HGCal_x_vs_z_vs_Rloc","HGCal_l_vs_z_vs_Rloc"}

_allmaterialsplots = {"HGCal_x_vs_eta","HGCal_l_vs_eta","HGCal_x_vs_phi","HGCal_l_vs_phi","HGCal_x_vs_R","HGCal_l_vs_R","HGCal_x_vs_eta_vs_phi","HGCal_l_vs_eta_vs_phi","HGCal_x_vs_z_vs_Rsum","HGCal_l_vs_z_vs_Rsum","HGCal_x_vs_z_vs_Rsumcos","HGCal_l_vs_z_vs_Rsumcos","HGCal_x_vs_z_vs_Rloc","HGCal_l_vs_z_vs_Rloc"}

_fromvertexplots = {"HGCal_l_vs_eta","HGCal_l_vs_z_vs_Rsum","HGCal_l_vs_z_vs_Rsum_Zpluszoom"}

_individualMatPlotsDesc = {
"HGCal_x_vs_z_vs_Rsum" : "The plots below shows the 2D profile histogram for THEMAT in all HGCAL that displays the mean value of the material budget in units of radiation length in each R-z cell. R-z cell is 1 cm x 1 mm. The plots depict the accumulated material budget as seen by the track, as the track travels throughout the detector.",
"HGCal_l_vs_z_vs_Rsum" : "The plots below shows the 2D profile histogram for THEMAT in all HGCAL that displays the mean value of the material budget in units of interaction length in each R-z cell. R-z cell is 1 cm x 1 mm. The plots depict the accumulated material budget as seen by the track, as the track travels throughout the detector.",
"HGCal_x_vs_z_vs_Rsumcos" : "The plots below shows the 2D profile histogram for THEMAT in all HGCAL that displays the mean value of the material budget in units of radiation length in each R-z cell. R-z cell is 1 cm x 1 mm. The plots depict the orthogonal accumulated material budget, that is cos(theta) what the track sees. ",
"HGCal_l_vs_z_vs_Rsumcos" : "The plots below shows the 2D profile histogram for THEMAT in all HGCAL that displays the mean value of the material budget in units of interaction length in each R-z cell. R-z cell is 1 cm x 1 mm. The plots depict the orthogonal accumulated material budget, that is cos(theta) what the track sees. ",
"HGCal_x_vs_z_vs_Rloc" : "The plots below shows the 2D profile histogram for THEMAT in all HGCAL that displays the local mean value of the material budget in units of radiation length in each R-z cell. R-z cell is 1 cm x 1 mm. The plots depict the local material budget as seen by the track, as the track travels throughout the detector. ",
"HGCal_l_vs_z_vs_Rloc" : "The plots below shows the 2D profile histogram for THEMAT in all HGCAL that displays the local mean value of the material budget in units of interaction length in each R-z cell. R-z cell is 1 cm x 1 mm. The plots depict the local material budget as seen by the track, as the track travels throughout the detector. "
}

_allmaterialsPlotsDesc= {
    "HGCal_x_vs_eta" : "The plot on the left shows the stacked profile histograms of all materials in HGCal geometry. These profile histograms display the mean value of the material budget in units of radiation length in each eta bin. 250 bins in eta (-5,5), so eta is divided in 0.04 width bins. ",

    "HGCal_l_vs_eta" : "The plot on the left shows the stacked profile histograms of all materials in HGCal geometry. These profile histograms display the mean value of the material budget in units of interaction length in each eta bin. 250 bins in eta (-5,5), so eta is divided in 0.04 width bins. ",

    "HGCal_x_vs_phi" : "The plot on the left shows the stacked profile histograms of all materials in HGCal geometry. These profile histograms display the mean value of the material budget in units of radiation length in each phi bin. 180 bins in phi (-3.2,3.2), so phi is divided in 0.036 rad width bins or 2.038 degrees width bins. ",

    "HGCal_l_vs_phi" : "The plot on the left shows the stacked profile histograms of all materials in HGCal geometry. These profile histograms display the mean value of the material budget in units of interaction length in each phi bin. 180 bins in phi -3.2,3.2), so phi is divided in 0.036 rad width bins or 2.038 degrees width bins. ",

    "HGCal_x_vs_R" : "The plot on the left shows the stacked profile histograms of all materials in HGCal geometry. These profile histograms display the mean value of the material budget in units of radiation length in each radius bin. 300 bins in radius (0,3000 mm), so radius is defined in 1 cm width bins. Both endcaps are in this histogram. Entries are huge since the radius is filled for each step of the track. Statistics in the HEB part above 1565 mm is smaller (although non visible, error is small), since in most part nothing is infront to keep account of the step. ",

    "HGCal_l_vs_R" : "The plot on the left shows the stacked profile histograms of all materials in HGCal geometry. These profile histograms display the mean value of the material budget in units of interaction length in each radius bin. 300 bins in radius (0,3000 mm), so radius is defined in 1 cm width bins. Both endcaps are in this histogram. Entries are huge since the radius is filled for each step of the track. Statistics in the HEB part above 1565 mm is smaller (although non visible, error is small), since in most part nothing is in front to keep account of the step. ", 

    "HGCal_x_vs_eta_vs_phi" : "The plot on the left shows the 2D profile histogram that displays the mean value of the material budget in units of radiation length in each eta-phi cell. 180 bins in phi (-3.2,3.2), so phi is divided in 0.036 rad width bins or 2.038 degrees width bins. 250 bins in eta -5., 5., so eta is divided in 0.04 width bins. Therefore, eta-phi cell is 2.038 degrees x 0.04 . ",

    "HGCal_l_vs_eta_vs_phi" : "The plot on the left shows the 2D profile histogram that displays the mean value of the material budget in units of interaction length in each eta-phi cell. 180 bins in phi (-3.2,3.2), so phi is divided in 0.036 rad width bins or 2.038 degrees width bins. 250 bins in eta -5., 5., so eta is divided in 0.04 width bins. Therefore, eta-phi cell is 2.038 degrees x 0.04 . ",
    
    "HGCal_x_vs_z_vs_Rsum" : "The plots below shows the 2D profile histogram that displays the mean value of the material budget in units of radiation length in each R-z cell. R-z cell is 1 cm x 1 mm. The plots depict the accumulated material budget as seen by the track, as the track travels throughout the detector.",
    
    "HGCal_l_vs_z_vs_Rsum" : "The plots below shows the 2D profile histogram that displays the mean value of the material budget in units of interaction length in each R-z cell. R-z cell is 1 cm x 1 mm. The plots depict the accumulated material budget as seen by the track, as the track travels throughout the detector.",
    
    "HGCal_x_vs_z_vs_Rsumcos" : "The plots below shows the 2D profile histogram that displays the mean value of the material budget in units of radiation length in each R-z cell. R-z cell is 1 cm x 1 mm. The plots depict the orthogonal accumulated material budget, that is cos(theta) what the track sees. ",
    
    "HGCal_l_vs_z_vs_Rsumcos" : "The plots below shows the 2D profile histogram that displays the mean value of the material budget in units of interaction length in each R-z cell. R-z cell is 1 cm x 1 mm. The plots depict the orthogonal accumulated material budget, that is cos(theta) what the track sees. " ,   
    
    "HGCal_x_vs_z_vs_Rloc" : "The plots below shows the 2D profile histogram that displays the local mean value of the material budget in units of radiation length in each R-z cell. R-z cell is 1 cm x 1 mm. The plots depict the local material budget as seen by the track, as the track travels throughout the detector. ",
    
    "HGCal_l_vs_z_vs_Rloc" : "The plots below shows the 2D profile histogram that displays the local mean value of the material budget in units of interaction length in each R-z cell. R-z cell is 1 cm x 1 mm. The plots depict the local material budget as seen by the track, as the track travels throughout the detector. "


}

_fromVertexPlotsDesc = {
   "HGCal_x_vs_eta" : "The plot below shows the stacked profile histogram of all sub detectors in front of muon stations. This profile histogram displays the mean value of the material budget in units of radiation length in each eta bin. 250 bins in eta (-5,5), so eta is divided in 0.04 width bins. ",
   
   "HGCal_l_vs_eta" : "The plots below shows the stacked profile histogram of all sub detectors in front of muon stations. This profile histogram displays the mean value of the material budget in units of interaction length in each eta bin. 250 bins in eta (-5,5), so eta is divided in 0.04 width bins. ",

   "HGCal_l_vs_z_vs_Rsum" : "The plots below shows the detectors that are taken into account in the calculation of the material budget. Keep in mind that coloured regions that depicts each sub-detector area may contain Air as material.",

   "HGCal_l_vs_z_vs_Rsum_Zpluszoom" : "The zoomed plots below shows the detectors that are taken into account in the calculation of the material budget. Keep in mind that coloured regions that depicts each sub-detector area may contain Air as material."
   


}

_hideShowFun = { 
     "thestyle" : "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"> \n <style> \n body {font-family: Arial;} \n.tab { \n  overflow: hidden; \n  border: 1px solid #ccc; \n  background-color: #f1f1f1;} \n .tab button {  background-color: inherit; \n  float: left; \n  border: none; \n  outline: none; \n  cursor: pointer; \n  padding: 14px 16px; \n  transition: 0.3s; \n  font-size: 17px; } \n .tab button:hover {  background-color: #ddd; } \n .tab button.active {  background-color: #ccc; } \n .tabcontent {  display: none; \n  padding: 6px 12px; \n  border: 1px solid #ccc; \n  border-top: none; \n} \n </style>",
     "buttonandFunction" : "<script> \n function openRegion(evt, regionName) { \n  var i, tabcontent, tablinks;\n  tabcontent = document.getElementsByClassName(\"tabcontent\"); \n  for (i = 0; i < tabcontent.length; i++) {\n    tabcontent[i].style.display = \"none\";\n  }\n  tablinks = document.getElementsByClassName(\"tablinks\"); \n  for (i = 0; i < tablinks.length; i++) {\n    tablinks[i].className = tablinks[i].className.replace(\" active\", \"\"); \n  }\n  document.getElementById(regionName).style.display = \"block\";\n  evt.currentTarget.className += \" active\"; \n}\n</script>\n",
     "divTabs" : "<div class=\"tab\">\n   <button class=\"tablinks\" onclick=\"openRegion(event, \'_AllHGCAL\')\">All HGCAL</button>\n   <button class=\"tablinks\" onclick=\"openRegion(event, \'_ZminusZoom\')\">Zminus</button>\n   <button class=\"tablinks\" onclick=\"openRegion(event, \'_ZplusZoom\')\">Zplus</button>\n </div>\n "
} 







