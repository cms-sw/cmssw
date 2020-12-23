import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester
from Configuration.Eras.Modifier_fastSim_cff import fastSim

def _addNoFlow(module):
    _noflowSeen = set()
    for eff in module.efficiency.value():
        tmp = eff.split(" ")
        if "cut" in tmp[0]:
            continue
        ind = -1
        if tmp[ind] == "fake" or tmp[ind] == "simpleratio":
            ind = -2
        if not tmp[ind] in _noflowSeen:
            module.noFlowDists.append(tmp[ind])
        if not tmp[ind-1] in _noflowSeen:
            module.noFlowDists.append(tmp[ind-1])

_defaultSubdirs = ["Tracking/Track/*", "Tracking/TrackTPPtLess09/*", "Tracking/TrackFromPV/*", "Tracking/TrackFromPVAllTP/*", "Tracking/TrackAllTPEffic/*", "Tracking/TrackBuilding/*","Tracking/TrackConversion/*", "Tracking/TrackGsf/*"]
_defaultSubdirsSummary = [e.replace("/*","") for e in _defaultSubdirs]

postProcessorTrack = DQMEDHarvester("DQMGenericClient",
    subDirs = cms.untracked.vstring(_defaultSubdirs),
    efficiency = cms.vstring(
    "effic 'Efficiency vs #eta' num_assoc(simToReco)_eta num_simul_eta",
    "efficPt 'Efficiency vs p_{T}' num_assoc(simToReco)_pT num_simul_pT",
#    "efficPtvseta 'Efficiency in p_{T}-#eta plane' num_assoc(simToReco)_pTvseta num_simul_pTvseta",
    "effic_vs_hit 'Efficiency vs hit' num_assoc(simToReco)_hit num_simul_hit",
    "effic_vs_layer 'Efficiency vs layer' num_assoc(simToReco)_layer num_simul_layer",
    "effic_vs_pixellayer 'Efficiency vs pixel layer' num_assoc(simToReco)_pixellayer num_simul_pixellayer",
    "effic_vs_3Dlayer 'Efficiency vs 3D layer' num_assoc(simToReco)_3Dlayer num_simul_3Dlayer",
    "effic_vs_pu 'Efficiency vs pu' num_assoc(simToReco)_pu num_simul_pu",
    "effic_vs_phi 'Efficiency vs #phi' num_assoc(simToReco)_phi num_simul_phi",
    "effic_vs_dxy 'Efficiency vs Dxy' num_assoc(simToReco)_dxy num_simul_dxy",
    "effic_vs_dz 'Efficiency vs Dz' num_assoc(simToReco)_dz num_simul_dz",
    "effic_vs_dxypv 'Efficiency vs Dxy(PV)' num_assoc(simToReco)_dxypv num_simul_dxypv",
    "effic_vs_dzpv 'Efficiency vs Dz(PV)' num_assoc(simToReco)_dzpv num_simul_dzpv",
    "effic_vs_dxypv_zoomed 'Efficiency vs Dxy(PV)' num_assoc(simToReco)_dxypv_zoomed num_simul_dxypv_zoomed",
    "effic_vs_dzpv_zoomed 'Efficiency vs Dz(PV)' num_assoc(simToReco)_dzpv_zoomed num_simul_dzpv_zoomed",
    "duplicatesRate 'Duplicates Rate vs #eta' num_duplicate_eta num_reco_eta",
    "duplicatesRate_Pt 'Duplicates Rate vs p_{T}' num_duplicate_pT num_reco_pT",
#    "duplicatesRate_Ptvseta 'Duplicates Rate in (p_{T}-#eta) plane' num_duplicate_pTvseta num_reco_pTvseta",
    "duplicatesRate_hit 'Duplicates Rate vs hit' num_duplicate_hit num_reco_hit",
    "duplicatesRate_layer 'Duplicates Rate vs layer' num_duplicate_layer num_reco_layer",
    "duplicatesRate_pixellayer 'Duplicates Rate vs pixel layer' num_duplicate_pixellayer num_reco_pixellayer",
    "duplicatesRate_3Dlayer 'Duplicates Rate vs layer' num_duplicate_3Dlayer num_reco_3Dlayer",
    "duplicatesRate_pu 'Duplicates Rate vs pu' num_duplicate_pu num_reco_pu",
    "duplicatesRate_phi 'Duplicates Rate vs #phi' num_duplicate_phi num_reco_phi",
    "duplicatesRate_dxy 'Duplicates Rate vs Dxy' num_duplicate_dxy num_reco_dxy",
    "duplicatesRate_dz 'Duplicates Rate vs Dz' num_duplicate_dz num_reco_dz",
    "duplicatesRate_dxypv 'Duplicates Rate vs Dxy(PV)' num_duplicate_dxypv num_reco_dxypv",
    "duplicatesRate_dzpv 'Duplicates Rate vs Dz(PV)' num_duplicate_dzpv num_reco_dzpv",
    "duplicatesRate_dxypv_zoomed 'Duplicates Rate vs Dxy(PV)' num_duplicate_dxypv_zoomed num_reco_dxypv_zoomed",
    "duplicatesRate_dzpv_zoomed 'Duplicates Rate vs Dz(PV)' num_duplicate_dzpv_zoomed num_reco_dzpv_zoomed",
    "duplicatesRate_vertpos 'Duplicates Rate vs vertpos' num_duplicate_vertpos num_reco_vertpos",
    "duplicatesRate_zpos 'Duplicates Rate vs zpos' num_duplicate_zpos num_reco_zpos",
    "duplicatesRate_dr 'Duplicates Rate vs dr' num_duplicate_dr num_reco_dr",
    "duplicatesRate_drj 'Duplicates Rate vs dr (track,jet)' num_duplicate_drj num_reco_drj",
    "duplicatesRate_chi2 'Duplicates Rate vs normalized #chi^{2}' num_duplicate_chi2 num_reco_chi2",
    "duplicatesRate_seedingLayerSet 'Duplicates rate vs. seedingLayerSet' num_duplicate_seedingLayerSet num_reco_seedingLayerSet",
    "chargeMisIdRate 'Charge MisID Rate vs #eta' num_chargemisid_eta num_reco_eta",
#    "chargeMisIdRate_Ptvseta 'Charge MisID Rate in (p_{T}-#eta) plane' num_chargemisid_pTvseta num_reco_pTvseta",
    "chargeMisIdRate_hit 'Charge MisID Rate vs hit' num_chargemisid_hit num_reco_hit",
    "chargeMisIdRate_layer 'Charge MisID Rate vs layer' num_chargemisid_hit num_reco_layer",
    "chargeMisIdRate_pixellayer 'Charge MisID Rate vs pixel layer' num_chargemisid_hit num_reco_pixellayer",
    "chargeMisIdRate_3Dlayer 'Charge MisID Rate vs 3Dlayer' num_chargemisid_hit num_reco_3Dlayer",
    "chargeMisIdRate_pu 'Charge MisID Rate vs pu' num_chargemisid_pu num_reco_pu",
    "chargeMisIdRate_phi 'Charge MisID Rate vs #phi' num_chargemisid_phi num_reco_phi",
    "chargeMisIdRate_dxy 'Charge MisID Rate vs Dxy' num_chargemisid_dxy num_reco_dxy",
    "chargeMisIdRate_dz 'Charge MisID Rate vs Dz' num_chargemisid_versus_dz num_reco_dz",
    "chargeMisIdRate_dxypv 'Charge MisID Rate vs Dxy(PV)' num_chargemisid_dxypv num_reco_dxypv",
    "chargeMisIdRate_dzpv 'Charge MisID Rate vs Dz(PV)' num_chargemisid_versus_dzpv num_reco_dzpv",
    "chargeMisIdRate_dxypv_zoomed 'Charge MisID Rate vs Dxy(PV)' num_chargemisid_dxypv_zoomed num_reco_dxypv_zoomed",
    "chargeMisIdRate_dzpv_zoomed 'Charge MisID Rate vs Dz(PV)' num_chargemisid_versus_dzpv_zoomed num_reco_dzpv_zoomed",
    "chargeMisIdRate_chi2 'Charge MisID Rate vs normalized #chi^{2}' num_chargemisid_chi2 num_reco_chi2",
    "effic_vs_vertpos 'Efficiency vs vertpos' num_assoc(simToReco)_vertpos num_simul_vertpos",
    "effic_vs_zpos 'Efficiency vs zpos' num_assoc(simToReco)_zpos num_simul_zpos",
    "effic_vs_dr 'Efficiency vs dr' num_assoc(simToReco)_dr num_simul_dr",
    "effic_vs_drj 'Efficiency vs dr (track,jet)' num_assoc(simToReco)_drj num_simul_drj",
    "effic_vertcount_barrel 'efficiency in barrel vs N of pileup vertices' num_assoc(simToReco)_vertcount_barrel num_simul_vertcount_barrel",
    "effic_vertcount_fwdpos 'efficiency in endcap(+) vs N of pileup vertices' num_assoc(simToReco)_vertcount_fwdpos num_simul_vertcount_fwdpos",
    "effic_vertcount_fwdneg 'efficiency in endcap(-) vs N of pileup vertices' num_assoc(simToReco)_vertcount_fwdneg num_simul_vertcount_fwdneg",
    "effic_vertz_barrel 'efficiency in barrel vs z of primary interaction vertex' num_assoc(simToReco)_vertz_barrel num_simul_vertz_barrel",
    "effic_vertz_fwdpos 'efficiency in endcap(+) vs z of primary interaction vertex' num_assoc(simToReco)_vertz_fwdpos num_simul_vertz_fwdpos",
    "effic_vertz_fwdneg 'efficiency in endcap(-) vs z of primary interaction vertex' num_assoc(simToReco)_vertz_fwdneg num_simul_vertz_fwdneg",
    "pileuprate 'Pileup Rate vs #eta' num_pileup_eta num_reco_eta",
    "pileuprate_Pt 'Pileup rate vs p_{T}' num_pileup_pT num_reco_pT",
#    "pileuprate_Ptvseta 'Pileup rate in (p_{T}-#eta) plane' num_pileup_pTvseta num_reco_pTvseta",
    "pileuprate_hit 'Pileup rate vs hit' num_pileup_hit num_reco_hit",
    "pileuprate_layer 'Pileup rate vs layer' num_pileup_layer num_reco_layer",
    "pileuprate_pixellayer 'Pileup rate vs layer' num_pileup_pixellayer num_reco_pixellayer",
    "pileuprate_3Dlayer 'Pileup rate vs 3D layer' num_pileup_3Dlayer num_reco_3Dlayer",
    "pileuprate_pu 'Pileup rate vs pu' num_pileup_pu num_reco_pu",
    "pileuprate_phi 'Pileup rate vs #phi' num_pileup_phi num_reco_phi",
    "pileuprate_dxy 'Pileup rate vs dxy' num_pileup_dxy num_reco_dxy",
    "pileuprate_dz 'Pileup rate vs dz' num_pileup_dz num_reco_dz",
    "pileuprate_dxypv 'Pileup rate vs dxy(PV)' num_pileup_dxypv num_reco_dxypv",
    "pileuprate_dzpv 'Pileup rate vs dz(PV)' num_pileup_dzpv num_reco_dzpv",
    "pileuprate_dxypv_zoomed 'Pileup rate vs dxy(PV)' num_pileup_dxypv_zoomed num_reco_dxypv_zoomed",
    "pileuprate_dzpv_zoomed 'Pileup rate vs dz(PV)' num_pileup_dzpv_zoomed num_reco_dzpv_zoomed",
    "pileuprate_vertpos 'Pileup rate vs vertpos' num_pileup_vertpos num_reco_vertpos",
    "pileuprate_zpos 'Pileup rate vs zpos' num_pileup_zpos num_reco_zpos",
    "pileuprate_dr 'Pileup rate vs dr' num_pileup_dr num_reco_dr",
    "pileuprate_drj 'Pileup rate vs dr (track,jet)' num_pileup_drj num_reco_drj",
    "pileuprate_chi2 'Pileup rate vs normalized #chi^{2}' num_pileup_chi2 num_reco_chi2",
    "pileuprate_seedingLayerSet 'Pileup rate vs. seedingLayerSet' num_pileup_seedingLayerSet num_reco_seedingLayerSet",
    "fakerate 'Fake rate vs #eta' num_assoc(recoToSim)_eta num_reco_eta fake",
    "fakeratePt 'Fake rate vs p_{T}' num_assoc(recoToSim)_pT num_reco_pT fake",
#    "fakeratePtvseta 'Fake rate in (p_{T}-#eta) plane' num_assoc(recoToSim)_pTvseta num_reco_pTvseta fake",
    "fakerate_vs_hit 'Fake rate vs hit' num_assoc(recoToSim)_hit num_reco_hit fake",
    "fakerate_vs_layer 'Fake rate vs layer' num_assoc(recoToSim)_layer num_reco_layer fake",
    "fakerate_vs_pixellayer 'Fake rate vs layer' num_assoc(recoToSim)_pixellayer num_reco_pixellayer fake",
    "fakerate_vs_3Dlayer 'Fake rate vs 3D layer' num_assoc(recoToSim)_3Dlayer num_reco_3Dlayer fake",
    "fakerate_vs_pu 'Fake rate vs pu' num_assoc(recoToSim)_pu num_reco_pu fake",
    "fakerate_vs_phi 'Fake rate vs phi' num_assoc(recoToSim)_phi num_reco_phi fake",
    "fakerate_vs_dxy 'Fake rate vs dxy' num_assoc(recoToSim)_dxy num_reco_dxy fake",
    "fakerate_vs_dz 'Fake rate vs dz' num_assoc(recoToSim)_dz num_reco_dz fake",
    "fakerate_vs_dxypv 'Fake rate vs dxypv' num_assoc(recoToSim)_dxypv num_reco_dxypv fake",
    "fakerate_vs_dzpv 'Fake rate vs dzpv' num_assoc(recoToSim)_dzpv num_reco_dzpv fake",
    "fakerate_vs_dxypv_zoomed 'Fake rate vs dxypv' num_assoc(recoToSim)_dxypv_zoomed num_reco_dxypv_zoomed fake",
    "fakerate_vs_dzpv_zoomed 'Fake rate vs dzpv' num_assoc(recoToSim)_dzpv_zoomed num_reco_dzpv_zoomed fake",
    "fakerate_vs_vertpos 'Fake rate vs vertpos' num_assoc(recoToSim)_vertpos num_reco_vertpos fake",
    "fakerate_vs_zpos 'Fake rate vs vertpos' num_assoc(recoToSim)_zpos num_reco_zpos fake",
    "fakerate_vs_dr 'Fake rate vs dr' num_assoc(recoToSim)_dr num_reco_dr fake",
    "fakerate_vs_drj 'Fake rate vs dr (track,jet)' num_assoc(recoToSim)_drj num_reco_drj fake",
    "fakerate_vs_chi2 'Fake rate vs normalized #chi^{2}' num_assoc(recoToSim)_chi2 num_reco_chi2 fake",
    "fakerate_vs_seedingLayerSet 'Fake rate vs. seedingLayerSet' num_assoc(recoToSim)_seedingLayerSet num_reco_seedingLayerSet fake",
    "fakerate_vertcount_barrel 'fake rate in barrel vs N of pileup vertices' num_assoc(recoToSim)_vertcount_barrel num_reco_vertcount_barrel fake",
    "fakerate_vertcount_fwdpos 'fake rate in endcap(+) vs N of pileup vertices' num_assoc(recoToSim)_vertcount_fwdpos num_reco_vertcount_fwdpos fake",
    "fakerate_vertcount_fwdneg 'fake rate in endcap(-) vs N of pileup vertices' num_assoc(recoToSim)_vertcount_fwdneg num_reco_vertcount_fwdneg fake"
    "fakerate_ootpu_entire 'fake rate from out of time pileup vs N of pileup vertices' num_assoc(recoToSim)_ootpu_entire num_reco_ootpu_entire",
    "fakerate_ootpu_barrel 'fake rate from out of time pileup in barrel vs N of pileup vertices' num_assoc(recoToSim)_ootpu_barrel num_reco_ootpu_barrel",
    "fakerate_ootpu_fwdpos 'fake rate from out of time pileup in endcap(+) vs N of pileup vertices' num_assoc(recoToSim)_ootpu_fwdpos num_reco_ootpu_fwdpos",
    "fakerate_ootpu_fwdneg 'fake rate from out of time pileup in endcap(-) vs N of pileup vertices' num_assoc(recoToSim)_ootpu_fwdneg num_reco_ootpu_fwdneg",

    "effic_vs_dzpvcut 'Efficiency vs. dz (PV)' num_assoc(simToReco)_dzpvcut num_simul_dzpvcut",
    "effic_vs_dzpvcut2 'Efficiency (tracking eff factorized out) vs. dz (PV)' num_assoc(simToReco)_dzpvcut num_simul2_dzpvcut",
    "fakerate_vs_dzpvcut 'Fake rate vs. dz(PV)' num_assoc(recoToSim)_dzpvcut num_reco_dzpvcut fake",
    "pileuprate_dzpvcut 'Pileup rate vs. dz(PV)' num_pileup_dzpvcut num_reco_dzpvcut",

    "effic_vs_dzpvcut_pt 'Fraction of true p_{T} carried by recoed TPs from PV vs. dz(PV)' num_assoc(simToReco)_dzpvcut_pt num_simul_dzpvcut_pt",
    "effic_vs_dzpvcut2_pt 'Fraction of true p_{T} carried by recoed TPs from PV (tracking eff factorized out) vs. dz(PV)' num_assoc(simToReco)_dzpvcut_pt num_simul2_dzpvcut_pt",
    "fakerate_vs_dzpvcut_pt 'Fraction of fake p_{T} carried by tracks from PV vs. dz(PV)' num_assoc(recoToSim)_dzpvcut_pt num_reco_dzpvcut_pt fake",
    "pileuprate_dzpvcut_pt 'Fraction of pileup p_{T} carried by tracks from PV vs. dz(PV)' num_pileup_dzpvcut_pt num_reco_dzpvcut_pt",

    "effic_vs_dzpvsigcut 'Efficiency vs. dz(PV)/dzError' num_assoc(simToReco)_dzpvsigcut num_simul_dzpvsigcut",
    "effic_vs_dzpvsigcut2 'Efficiency (tracking eff factorized out) vs. dz(PV)/dzError' num_assoc(simToReco)_dzpvsigcut num_simul2_dzpvsigcut",
    "fakerate_vs_dzpvsigcut 'Fake rate vs. dz(PV)/dzError' num_assoc(recoToSim)_dzpvsigcut num_reco_dzpvsigcut fake",
    "pileuprate_dzpvsigcut 'Pileup rate vs. dz(PV)/dzError' num_pileup_dzpvsigcut num_reco_dzpvsigcut",

    "effic_vs_dzpvsigcut_pt 'Fraction of true p_{T} carried by recoed TPs from PV vs. dz(PV)/dzError' num_assoc(simToReco)_dzpvsigcut_pt num_simul_dzpvsigcut_pt",
    "effic_vs_dzpvsigcut2_pt 'Fraction of true p_{T} carried by recoed TPs from PV (tracking eff factorized out) vs. dz(PV)/dzError' num_assoc(simToReco)_dzpvsigcut_pt num_simul2_dzpvsigcut_pt",
    "fakerate_vs_dzpvsigcut_pt 'Fraction of fake p_{T} carried by tracks from PV vs. dz(PV)/dzError' num_assoc(recoToSim)_dzpvsigcut_pt num_reco_dzpvsigcut_pt fake",
    "pileuprate_dzpvsigcut_pt 'Fraction of pileup p_{T} carried by tracks from PV vs. dz(PV)/dzError' num_pileup_dzpvsigcut_pt num_reco_dzpvsigcut_pt",

    "effic_vs_simpvz 'Efficiency vs. sim PV z' num_assoc(simToReco)_simpvz num_simul_simpvz",
    "fakerate_vs_simpvz 'Fake rate vs. sim PV z' num_assoc(recoToSim)_simpvz num_reco_simpvz fake",
    "duplicatesRate_simpvz 'Duplicates Rate vs sim PV z' num_duplicate_simpvz num_reco_simpvz",
    "pileuprate_simpvz 'Pileup rate vs. sim PV z' num_pileup_simpvz num_reco_simpvz",

    "fakerate_vs_mva1 'Fake rate vs. MVA1' num_assoc(recoToSim)_mva1 num_reco_mva1 fake",
    "fakerate_vs_mva2 'Fake rate vs. MVA2' num_assoc(recoToSim)_mva2 num_reco_mva2 fake",
    "fakerate_vs_mva3 'Fake rate vs. MVA3' num_assoc(recoToSim)_mva3 num_reco_mva3 fake",

    "effic_vs_mva1cut 'Efficiency (tracking eff factorized out) vs. MVA1' num_assoc(simToReco)_mva1cut num_simul2_mva1cut",
    "fakerate_vs_mva1cut 'Fake rate vs. MVA1' num_assoc(recoToSim)_mva1cut num_reco_mva1cut fake",
    "effic_vs_mva2cut 'Efficiency (tracking eff factorized out) vs. MVA2' num_assoc(simToReco)_mva2cut num_simul2_mva2cut",
    "effic_vs_mva2cut_hp 'Efficiency (tracking eff factorized out) vs. MVA2' num_assoc(simToReco)_mva2cut_hp num_simul2_mva2cut_hp",
    "fakerate_vs_mva2cut 'Fake rate vs. MVA2' num_assoc(recoToSim)_mva2cut num_reco_mva2cut fake",
    "fakerate_vs_mva2cut_hp 'Fake rate vs. MVA2' num_assoc(recoToSim)_mva2cut_hp num_reco_mva2cut_hp fake",
    "effic_vs_mva3cut 'Efficiency (tracking eff factorized out) vs. MVA3' num_assoc(simToReco)_mva3cut num_simul2_mva3cut",
    "effic_vs_mva3cut_hp 'Efficiency (tracking eff factorized out) vs. MVA3' num_assoc(simToReco)_mva3cut_hp num_simul2_mva3cut_hp",
    "fakerate_vs_mva3cut 'Fake rate vs. MVA3' num_assoc(recoToSim)_mva3cut num_reco_mva3cut fake",
    "fakerate_vs_mva3cut_hp 'Fake rate vs. MVA3' num_assoc(recoToSim)_mva3cut_hp num_reco_mva3cut_hp fake",
    ),
    resolution = cms.vstring(
                             "cotThetares_vs_eta '#sigma(cot(#theta)) vs #eta' cotThetares_vs_eta",
                             "cotThetares_vs_pt '#sigma(cot(#theta)) vs p_{T}' cotThetares_vs_pt",
                             "h_dxypulleta 'd_{xy} Pull vs #eta' dxypull_vs_eta",
                             "h_dxypullpt 'd_{xy} Pull vs p_{T}' dxypull_vs_pt",
                             "dxyres_vs_eta '#sigma(d_{xy}) vs #eta' dxyres_vs_eta",
                             "dxyres_vs_phi '#sigma(d_{xy}) vs #phi' dxyres_vs_phi",
                             "dxyres_vs_pt '#sigma(d_{xy}) vs p_{T}' dxyres_vs_pt",
                             "h_dzpulleta 'd_{z} Pull vs #eta' dzpull_vs_eta",
                             "h_dzpullpt 'd_{z} Pull vs p_{T}' dzpull_vs_pt",
                             "dzres_vs_eta '#sigma(d_{z}) vs #eta' dzres_vs_eta",
                             "dzres_vs_phi '#sigma(d_{z}) vs #phi' dzres_vs_phi",
                             "dzres_vs_pt '#sigma(d_{z}) vs p_{T}' dzres_vs_pt",
                             "etares_vs_eta '#sigma(#eta) vs #eta' etares_vs_eta",
                             "h_phipulleta '#phi Pull vs #eta' phipull_vs_eta",
                             "h_phipullpt '#phi Pull vs p_{T}' phipull_vs_pt",
                             "h_phipullphi '#phi Pull vs #phi' phipull_vs_phi",
                             "phires_vs_eta '#sigma(#phi) vs #eta' phires_vs_eta",
                             "phires_vs_phi '#sigma(#phi) vs #phi' phires_vs_phi",
                             "phires_vs_pt '#sigma(#phi) vs p_{T}' phires_vs_pt",
                             "h_ptpulleta 'p_{T} Pull vs #eta' ptpull_vs_eta",
                             "h_ptpullpt 'p_{T} Pull vs p_{T}' ptpull_vs_pt",
                             "h_ptpullphi 'p_{T} Pull vs #phi' ptpull_vs_phi",
                             "ptres_vs_eta '#sigma(p_{T}) vs #eta' ptres_vs_eta",
                             "ptres_vs_phi '#sigma(p_{T}) vs #phi' ptres_vs_phi",
                             "ptres_vs_pt '#sigma(p_{T}) vs p_{T}' ptres_vs_pt",
                             "h_thetapulleta '#theta Pull vs #eta' thetapull_vs_eta",
                             "h_thetapullpt '#theta Pull vs p_{T}' thetapull_vs_pt",
                             "h_thetapullphi '#theta Pull vs #phi' thetapull_vs_phi"
                             ),
    cumulativeDists = cms.untracked.vstring(
        "num_reco_dzpvcut",
        "num_assoc(recoToSim)_dzpvcut",
        "num_assoc(simToReco)_dzpvcut",
        "num_simul_dzpvcut",
        "num_simul2_dzpvcut",
        "num_pileup_dzpvcut",
        "num_reco_dzpvcut_pt",
        "num_assoc(recoToSim)_dzpvcut_pt",
        "num_assoc(simToReco)_dzpvcut_pt",
        "num_simul_dzpvcut_pt",
        "num_simul2_dzpvcut_pt",
        "num_pileup_dzpvcut_pt",
        "num_reco_dzpvsigcut",
        "num_assoc(recoToSim)_dzpvsigcut",
        "num_assoc(simToReco)_dzpvsigcut",
        "num_simul_dzpvsigcut",
        "num_simul2_dzpvsigcut",
        "num_pileup_dzpvsigcut",
        "num_reco_dzpvsigcut_pt",
        "num_assoc(recoToSim)_dzpvsigcut_pt",
        "num_assoc(simToReco)_dzpvsigcut_pt",
        "num_simul_dzpvsigcut_pt",
        "num_simul2_dzpvsigcut_pt",
        "num_pileup_dzpvsigcut_pt",
        "num_reco_mva1cut descending",
        "num_reco_mva2cut descending",
        "num_reco_mva2cut_hp descending",
        "num_reco_mva3cut descending",
        "num_reco_mva3cut_hp descending",
        "num_assoc(recoToSim)_mva1cut descending",
        "num_assoc(recoToSim)_mva2cut descending",
        "num_assoc(recoToSim)_mva2cut_hp descending",
        "num_assoc(recoToSim)_mva3cut descending",
        "num_assoc(recoToSim)_mva3cut_hp descending",
        "num_assoc(simToReco)_mva1cut descending",
        "num_assoc(simToReco)_mva2cut descending",
        "num_assoc(simToReco)_mva2cut_hp descending",
        "num_assoc(simToReco)_mva3cut descending",
        "num_assoc(simToReco)_mva3cut_hp descending",
        "num_simul2_mva1cut descending",
        "num_simul2_mva2cut descending",
        "num_simul2_mva2cut_hp descending",
        "num_simul2_mva3cut descending",
        "num_simul2_mva3cut_hp descending",
    ),
    noFlowDists = cms.untracked.vstring(),
    outputFileName = cms.untracked.string("")
)
_addNoFlow(postProcessorTrack)

postProcessorTrack2D = DQMEDHarvester("DQMGenericClient",
    makeGlobalEffienciesPlot = cms.untracked.bool(False),
    subDirs = cms.untracked.vstring(_defaultSubdirs),
    efficiency = cms.vstring(
    "efficPtvseta 'Efficiency in p_{T}-#eta plane' num_assoc(simToReco)_pTvseta num_simul_pTvseta",
    "duplicatesRate_Ptvseta 'Duplicates Rate in (p_{T}-#eta) plane' num_duplicate_pTvseta num_reco_pTvseta",
    "chargeMisIdRate_Ptvseta 'Charge MisID Rate in (p_{T}-#eta) plane' num_chargemisid_pTvseta num_reco_pTvseta",
    "pileuprate_Ptvseta 'Pileup rate in (p_{T}-#eta) plane' num_pileup_pTvseta num_reco_pTvseta",
    "fakeratePtvseta 'Fake rate in (p_{T}-#eta) plane' num_assoc(recoToSim)_pTvseta num_reco_pTvseta fake",
    ),
    resolution = cms.vstring(),
    noFlowDists = cms.untracked.vstring(),
    outputFileName = cms.untracked.string("")
)
_addNoFlow(postProcessorTrack2D)

# nrec/nsim makes sense only for
# - all tracks vs. all in-time TrackingParticles
# - PV tracks vs. signal TrackingParticles
postProcessorTrackNrecVsNsim = DQMEDHarvester("DQMGenericClient",
    subDirs = cms.untracked.vstring("Tracking/TrackFromPV/*", "Tracking/TrackAllTPEffic/*"),
    efficiency = cms.vstring(
        "nrecPerNsim 'Tracks/TrackingParticles vs #eta' num_reco2_eta num_simul_eta simpleratio",
        "nrecPerNsimPt 'Tracks/TrackingParticles vs p_{T}' num_reco2_pT num_simul_pT simpleratio",
#        "nrecPerNsimPtvseta 'Tracks/TrackingParticles in (p_{T}-#eta) plane' num_reco2_pTvseta num_simul_pTvseta simpleratio",
        "nrecPerNsim_vs_pu 'Tracks/TrackingParticles vs pu' num_reco2_pu num_simul_pu simpleratio",
    ),
    resolution = cms.vstring(),
    noFlowDists = cms.untracked.vstring(),
)
_addNoFlow(postProcessorTrackNrecVsNsim)
postProcessorTrackNrecVsNsim2D = DQMEDHarvester("DQMGenericClient",
    makeGlobalEffienciesPlot = cms.untracked.bool(False),
    subDirs = cms.untracked.vstring("Tracking/TrackFromPV/*", "Tracking/TrackAllTPEffic/*"),
    efficiency = cms.vstring(
        "nrecPerNsimPtvseta 'Tracks/TrackingParticles in (p_{T}-#eta) plane' num_reco2_pTvseta num_simul_pTvseta simpleratio",
    ),
    resolution = cms.vstring(),
    noFlowDists = cms.untracked.vstring(),
)
_addNoFlow(postProcessorTrackNrecVsNsim2D)


postProcessorTrackSummary = DQMEDHarvester("DQMGenericClient",
    subDirs = cms.untracked.vstring(_defaultSubdirsSummary),
    efficiency = cms.vstring(
    "effic_vs_coll 'Efficiency vs track collection' num_assoc(simToReco)_coll num_simul_coll",
    "effic_vs_coll_allPt 'Efficiency vs track collection' num_assoc(simToReco)_coll_allPt num_simul_coll_allPt",
    "duplicatesRate_coll 'Duplicates Rate vs track collection' num_duplicate_coll num_reco_coll",
    "pileuprate_coll 'Pileup rate vs track collection' num_pileup_coll num_reco_coll",
    "fakerate_vs_coll 'Fake rate vs track collection' num_assoc(recoToSim)_coll num_reco_coll fake",
    ),
    resolution = cms.vstring(),
    noFlowDists = cms.untracked.vstring(),
)
_addNoFlow(postProcessorTrackSummary)

postProcessorTrackSequence = cms.Sequence(
    postProcessorTrack+
    postProcessorTrackNrecVsNsim+
    postProcessorTrackSummary
)

fastSim.toModify(postProcessorTrack, subDirs = [e for e in _defaultSubdirs if e not in ["Tracking/TrackGsf/*","Tracking/TrackConversion/*"]])
fastSim.toModify(postProcessorTrackSummary, subDirs = [e for e in _defaultSubdirsSummary if e not in ["Tracking/TrackGsf","Tracking/TrackConversion"]])

#######
# Define a standalone seuquence to support the Standalone harvesting mode
# see https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideMultiTrackValidator#cmsDriver_MTV_alone_i_e_standalone for more information
########

postProcessorTrackStandalone = postProcessorTrack.clone(
    subDirs = _defaultSubdirs+["Tracking/TrackBHadron/*"]
)
postProcessorTrackSummaryStandalone = postProcessorTrackSummary.clone(
    subDirs = _defaultSubdirs+["Tracking/TrackBHadron"]
)

postProcessorTrackSequenceStandalone = cms.Sequence(
    postProcessorTrackStandalone+
    postProcessorTrackNrecVsNsim+
    postProcessorTrackSummaryStandalone
)

postProcessorTrackPhase2 = postProcessorTrack.clone()
postProcessorTrackPhase2.subDirs.extend(["Tracking/TrackTPEtaGreater2p7/*"])
postProcessorTrackSummaryPhase2 = postProcessorTrackSummary.clone()
postProcessorTrackSummaryPhase2.subDirs.extend(["Tracking/TrackTPEtaGreater2p7/*"])

from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
phase2_tracker.toReplaceWith(postProcessorTrack,postProcessorTrackPhase2)
phase2_tracker.toReplaceWith(postProcessorTrackSummary,postProcessorTrackSummaryPhase2)

postProcessorTrackTrackingOnly = postProcessorTrack.clone()
postProcessorTrackTrackingOnly.subDirs.extend(["Tracking/TrackBHadron/*", "Tracking/TrackSeeding/*", "Tracking/PixelTrack/*", "Tracking/PixelTrackFromPV/*", "Tracking/PixelTrackFromPVAllTP/*", "Tracking/PixelTrackBHadron/*"])
postProcessorTrackSummaryTrackingOnly = postProcessorTrackSummary.clone()
postProcessorTrackSummaryTrackingOnly.subDirs.extend(["Tracking/TrackBHadron", "Tracking/TrackSeeding", "Tracking/PixelTrack", "Tracking/PixelTrackFromPV", "Tracking/PixelTrackFromPVAllTP", "Tracking/PixelTrackBHadron"])

postProcessorTrackSequenceTrackingOnly = cms.Sequence(
    postProcessorTrackTrackingOnly+
    postProcessorTrackNrecVsNsim+
    postProcessorTrackSummaryTrackingOnly
)

fastSim.toModify(postProcessorTrackTrackingOnly,subDirs = [e for e in _defaultSubdirs if e not in ["Tracking/TrackGsf/*","Tracking/TrackConversion/*","Tracking/TrackBHadron/*"]])
fastSim.toModify(postProcessorTrackSummaryTrackingOnly,subDirs = [e for e in _defaultSubdirsSummary if e not in ["Tracking/TrackGsf","Tracking/TrackConversion","Tracking/TrackBHadron"]])
