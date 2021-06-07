
// user include files
#include "Validation/RecoEgamma/plugins/ElectronMcFakeValidator.h"

#include "RecoEgamma/EgammaElectronAlgos/interface/ElectronUtilities.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedFwd.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/ValueMap.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "TMath.h"
#include "TFile.h"
#include "TH1F.h"
#include "TH1I.h"
#include "TH2F.h"
#include "TProfile.h"
#include "TTree.h"
#include <vector>
#include <iostream>

using namespace reco;

ElectronMcFakeValidator::ElectronMcFakeValidator(const edm::ParameterSet &conf) : ElectronDqmAnalyzerBase(conf) {
  electronCollection_ = consumes<reco::GsfElectronCollection>(conf.getParameter<edm::InputTag>("electronCollection"));
  electronCoreCollection_ =
      consumes<reco::GsfElectronCoreCollection>(conf.getParameter<edm::InputTag>("electronCoreCollection"));
  electronTrackCollection_ =
      consumes<reco::GsfTrackCollection>(conf.getParameter<edm::InputTag>("electronTrackCollection"));
  electronSeedCollection_ =
      consumes<reco::ElectronSeedCollection>(conf.getParameter<edm::InputTag>("electronSeedCollection"));
  matchingObjectCollection_ =
      consumes<reco::GenJetCollection>(conf.getParameter<edm::InputTag>("matchingObjectCollection"));
  offlineVerticesCollection_ =
      consumes<reco::VertexCollection>(conf.getParameter<edm::InputTag>("offlinePrimaryVertices"));

  beamSpotTag_ = consumes<reco::BeamSpot>(conf.getParameter<edm::InputTag>("beamSpot"));
  readAOD_ = conf.getParameter<bool>("readAOD");

  isoFromDepsTk03Tag_ = consumes<edm::ValueMap<double>>(conf.getParameter<edm::InputTag>("isoFromDepsTk03"));
  isoFromDepsTk04Tag_ = consumes<edm::ValueMap<double>>(conf.getParameter<edm::InputTag>("isoFromDepsTk04"));
  isoFromDepsEcalFull03Tag_ =
      consumes<edm::ValueMap<double>>(conf.getParameter<edm::InputTag>("isoFromDepsEcalFull03"));
  isoFromDepsEcalFull04Tag_ =
      consumes<edm::ValueMap<double>>(conf.getParameter<edm::InputTag>("isoFromDepsEcalFull04"));
  isoFromDepsEcalReduced03Tag_ =
      consumes<edm::ValueMap<double>>(conf.getParameter<edm::InputTag>("isoFromDepsEcalReduced03"));
  isoFromDepsEcalReduced04Tag_ =
      consumes<edm::ValueMap<double>>(conf.getParameter<edm::InputTag>("isoFromDepsEcalReduced04"));
  isoFromDepsHcal03Tag_ = consumes<edm::ValueMap<double>>(conf.getParameter<edm::InputTag>("isoFromDepsHcal03"));
  isoFromDepsHcal04Tag_ = consumes<edm::ValueMap<double>>(conf.getParameter<edm::InputTag>("isoFromDepsHcal04"));

  maxPt_ = conf.getParameter<double>("MaxPt");
  maxAbsEta_ = conf.getParameter<double>("MaxAbsEta");
  deltaR_ = conf.getParameter<double>("DeltaR");
  inputFile_ = conf.getParameter<std::string>("InputFile");
  outputFile_ = conf.getParameter<std::string>("OutputFile");
  inputInternalPath_ = conf.getParameter<std::string>("InputFolderName");
  outputInternalPath_ = conf.getParameter<std::string>("OutputFolderName");

  // histos bining and limits

  edm::ParameterSet histosSet = conf.getParameter<edm::ParameterSet>("histosCfg");

  xyz_nbin = histosSet.getParameter<int>("Nbinxyz");

  p_nbin = histosSet.getParameter<int>("Nbinp");
  p2D_nbin = histosSet.getParameter<int>("Nbinp2D");
  p_max = histosSet.getParameter<double>("Pmax");

  pt_nbin = histosSet.getParameter<int>("Nbinpt");
  pt2D_nbin = histosSet.getParameter<int>("Nbinpt2D");
  pteff_nbin = histosSet.getParameter<int>("Nbinpteff");
  pt_max = histosSet.getParameter<double>("Ptmax");

  fhits_nbin = histosSet.getParameter<int>("Nbinfhits");
  fhits_max = histosSet.getParameter<double>("Fhitsmax");

  lhits_nbin = histosSet.getParameter<int>("Nbinlhits");
  lhits_max = histosSet.getParameter<double>("Lhitsmax");

  eop_nbin = histosSet.getParameter<int>("Nbineop");
  eop2D_nbin = histosSet.getParameter<int>("Nbineop2D");
  eop_max = histosSet.getParameter<double>("Eopmax");
  eopmaxsht = histosSet.getParameter<double>("Eopmaxsht");

  eta_nbin = histosSet.getParameter<int>("Nbineta");
  eta2D_nbin = histosSet.getParameter<int>("Nbineta2D");
  eta_min = histosSet.getParameter<double>("Etamin");
  eta_max = histosSet.getParameter<double>("Etamax");

  deta_nbin = histosSet.getParameter<int>("Nbindeta");
  deta_min = histosSet.getParameter<double>("Detamin");
  deta_max = histosSet.getParameter<double>("Detamax");

  detamatch_nbin = histosSet.getParameter<int>("Nbindetamatch");
  detamatch2D_nbin = histosSet.getParameter<int>("Nbindetamatch2D");
  detamatch_min = histosSet.getParameter<double>("Detamatchmin");
  detamatch_max = histosSet.getParameter<double>("Detamatchmax");

  phi_nbin = histosSet.getParameter<int>("Nbinphi");
  phi2D_nbin = histosSet.getParameter<int>("Nbinphi2D");
  phi_min = histosSet.getParameter<double>("Phimin");
  phi_max = histosSet.getParameter<double>("Phimax");

  dphi_nbin = histosSet.getParameter<int>("Nbindphi");
  dphi_min = histosSet.getParameter<double>("Dphimin");
  dphi_max = histosSet.getParameter<double>("Dphimax");

  dphimatch_nbin = histosSet.getParameter<int>("Nbindphimatch");
  dphimatch2D_nbin = histosSet.getParameter<int>("Nbindphimatch2D");
  dphimatch_min = histosSet.getParameter<double>("Dphimatchmin");
  dphimatch_max = histosSet.getParameter<double>("Dphimatchmax");

  mee_nbin = histosSet.getParameter<int>("Nbinmee");
  mee_min = histosSet.getParameter<double>("Meemin");
  mee_max = histosSet.getParameter<double>("Meemax");

  hoe_nbin = histosSet.getParameter<int>("Nbinhoe");
  hoe_min = histosSet.getParameter<double>("Hoemin");
  hoe_max = histosSet.getParameter<double>("Hoemax");

  popmatching_nbin = histosSet.getParameter<int>("Nbinpopmatching");
  popmatching_min = histosSet.getParameter<double>("Popmatchingmin");
  popmatching_max = histosSet.getParameter<double>("Popmatchingmax");

  set_EfficiencyFlag = histosSet.getParameter<bool>("EfficiencyFlag");
  set_StatOverflowFlag = histosSet.getParameter<bool>("StatOverflowFlag");

  opv_nbin = histosSet.getParameter<int>("NbinOPV");
  opv_min = histosSet.getParameter<double>("OPV_min");
  opv_max = histosSet.getParameter<double>("OPV_max");

  ele_nbin = histosSet.getParameter<int>("NbinELE");
  ele_min = histosSet.getParameter<double>("ELE_min");
  ele_max = histosSet.getParameter<double>("ELE_max");

  core_nbin = histosSet.getParameter<int>("NbinCORE");
  core_min = histosSet.getParameter<double>("CORE_min");
  core_max = histosSet.getParameter<double>("CORE_max");

  track_nbin = histosSet.getParameter<int>("NbinTRACK");
  track_min = histosSet.getParameter<double>("TRACK_min");
  track_max = histosSet.getParameter<double>("TRACK_max");

  seed_nbin = histosSet.getParameter<int>("NbinSEED");
  seed_min = histosSet.getParameter<double>("SEED_min");
  seed_max = histosSet.getParameter<double>("SEED_max");

  // so to please coverity
  h1_matchingObjectNum = nullptr;
  h1_recEleNum_ = nullptr;
  h1_recCoreNum_ = nullptr;
  h1_recTrackNum_ = nullptr;
  h1_recSeedNum_ = nullptr;
  h1_recOfflineVertices_ = nullptr;

  h1_matchingObjectEta = nullptr;
  h1_matchingObjectAbsEta = nullptr;
  h1_matchingObjectP = nullptr;
  h1_matchingObjectPt = nullptr;
  h1_matchingObjectPhi = nullptr;
  h1_matchingObjectZ = nullptr;

  h1_ele_EoverP_all = nullptr;
  h1_ele_EseedOP_all = nullptr;
  h1_ele_EoPout_all = nullptr;
  h1_ele_EeleOPout_all = nullptr;
  h1_ele_dEtaSc_propVtx_all = nullptr;
  h1_ele_dPhiSc_propVtx_all = nullptr;
  h1_ele_dEtaCl_propOut_all = nullptr;
  h1_ele_dPhiCl_propOut_all = nullptr;
  h1_ele_TIP_all = nullptr;
  h1_ele_HoE_all = nullptr;
  h1_ele_vertexEta_all = nullptr;
  h1_ele_vertexPt_all = nullptr;
  h1_ele_mee_all = nullptr;
  h1_ele_mee_os = nullptr;

  h2_ele_E2mnE1vsMee_all = nullptr;
  h2_ele_E2mnE1vsMee_egeg_all = nullptr;

  h1_ele_matchingObjectEta_matched = nullptr;
  h1_ele_matchingObjectAbsEta_matched = nullptr;
  h1_ele_matchingObjectPt_matched = nullptr;
  h1_ele_matchingObjectPhi_matched = nullptr;
  h1_ele_matchingObjectZ_matched = nullptr;

  h1_ele_charge = nullptr;
  h2_ele_chargeVsEta = nullptr;
  h2_ele_chargeVsPhi = nullptr;
  h2_ele_chargeVsPt = nullptr;
  h1_ele_vertexP = nullptr;
  h1_ele_vertexPt = nullptr;
  h2_ele_vertexPtVsEta = nullptr;
  h2_ele_vertexPtVsPhi = nullptr;
  h1_ele_vertexEta = nullptr;
  h2_ele_vertexEtaVsPhi = nullptr;
  h1_ele_vertexAbsEta = nullptr;
  h1_ele_vertexPhi = nullptr;
  h1_ele_vertexX = nullptr;
  h1_ele_vertexY = nullptr;
  h1_ele_vertexZ = nullptr;
  h1_ele_vertexTIP = nullptr;
  h2_ele_vertexTIPVsEta = nullptr;
  h2_ele_vertexTIPVsPhi = nullptr;
  h2_ele_vertexTIPVsPt = nullptr;

  h1_ele_PoPmatchingObject = nullptr;
  h2_ele_PoPmatchingObjectVsEta = nullptr;
  h2_ele_PoPmatchingObjectVsPhi = nullptr;
  h2_ele_PoPmatchingObjectVsPt = nullptr;
  h1_ele_PoPmatchingObject_barrel = nullptr;
  h1_ele_PoPmatchingObject_endcaps = nullptr;

  h1_ele_EtaMnEtamatchingObject = nullptr;
  h2_ele_EtaMnEtamatchingObjectVsEta = nullptr;
  h2_ele_EtaMnEtamatchingObjectVsPhi = nullptr;
  h2_ele_EtaMnEtamatchingObjectVsPt = nullptr;
  h1_ele_PhiMnPhimatchingObject = nullptr;
  h1_ele_PhiMnPhimatchingObject2 = nullptr;
  h2_ele_PhiMnPhimatchingObjectVsEta = nullptr;
  h2_ele_PhiMnPhimatchingObjectVsPhi = nullptr;
  h2_ele_PhiMnPhimatchingObjectVsPt = nullptr;

  h1_scl_En_ = nullptr;
  h1_scl_EoEmatchingObject_barrel = nullptr;
  h1_scl_EoEmatchingObject_endcaps = nullptr;
  h1_scl_Et_ = nullptr;
  h2_scl_EtVsEta_ = nullptr;
  h2_scl_EtVsPhi_ = nullptr;
  h2_scl_EtaVsPhi_ = nullptr;
  h1_scl_Eta_ = nullptr;
  h1_scl_Phi_ = nullptr;

  h1_scl_SigIEtaIEta_ = nullptr;
  h1_scl_SigIEtaIEta_barrel_ = nullptr;
  h1_scl_SigIEtaIEta_endcaps_ = nullptr;
  h1_scl_full5x5_sigmaIetaIeta_ = nullptr;          // new 2014.01.12
  h1_scl_full5x5_sigmaIetaIeta_barrel_ = nullptr;   // new 2014.01.12
  h1_scl_full5x5_sigmaIetaIeta_endcaps_ = nullptr;  // new 2014.01.12
  h1_scl_E1x5_ = nullptr;
  h1_scl_E1x5_barrel_ = nullptr;
  h1_scl_E1x5_endcaps_ = nullptr;
  h1_scl_E2x5max_ = nullptr;
  h1_scl_E2x5max_barrel_ = nullptr;
  h1_scl_E2x5max_endcaps_ = nullptr;
  h1_scl_E5x5_ = nullptr;
  h1_scl_E5x5_barrel_ = nullptr;
  h1_scl_E5x5_endcaps_ = nullptr;

  h1_ele_ambiguousTracks = nullptr;
  h2_ele_ambiguousTracksVsEta = nullptr;
  h2_ele_ambiguousTracksVsPhi = nullptr;
  h2_ele_ambiguousTracksVsPt = nullptr;
  h1_ele_foundHits = nullptr;
  h1_ele_foundHits_barrel = nullptr;
  h1_ele_foundHits_endcaps = nullptr;
  h2_ele_foundHitsVsEta = nullptr;
  h2_ele_foundHitsVsPhi = nullptr;
  h2_ele_foundHitsVsPt = nullptr;
  h1_ele_lostHits = nullptr;
  h1_ele_lostHits_barrel = nullptr;
  h1_ele_lostHits_endcaps = nullptr;
  h2_ele_lostHitsVsEta = nullptr;
  h2_ele_lostHitsVsPhi = nullptr;
  h2_ele_lostHitsVsPt = nullptr;
  h1_ele_chi2 = nullptr;
  h1_ele_chi2_barrel = nullptr;
  h1_ele_chi2_endcaps = nullptr;
  h2_ele_chi2VsEta = nullptr;
  h2_ele_chi2VsPhi = nullptr;
  h2_ele_chi2VsPt = nullptr;

  h1_ele_PinMnPout = nullptr;
  h1_ele_PinMnPout_mode = nullptr;
  h2_ele_PinMnPoutVsEta_mode = nullptr;
  h2_ele_PinMnPoutVsPhi_mode = nullptr;
  h2_ele_PinMnPoutVsPt_mode = nullptr;
  h2_ele_PinMnPoutVsE_mode = nullptr;
  h2_ele_PinMnPoutVsChi2_mode = nullptr;

  h1_ele_outerP = nullptr;
  h1_ele_outerP_mode = nullptr;
  h2_ele_outerPVsEta_mode = nullptr;
  h1_ele_outerPt = nullptr;
  h1_ele_outerPt_mode = nullptr;
  h2_ele_outerPtVsEta_mode = nullptr;
  h2_ele_outerPtVsPhi_mode = nullptr;
  h2_ele_outerPtVsPt_mode = nullptr;
  h1_ele_EoP = nullptr;
  h1_ele_EoP_barrel = nullptr;
  h1_ele_EoP_endcaps = nullptr;
  h2_ele_EoPVsEta = nullptr;
  h2_ele_EoPVsPhi = nullptr;
  h2_ele_EoPVsE = nullptr;
  h1_ele_EseedOP = nullptr;
  h1_ele_EseedOP_barrel = nullptr;
  h1_ele_EseedOP_endcaps = nullptr;
  h2_ele_EseedOPVsEta = nullptr;
  h2_ele_EseedOPVsPhi = nullptr;
  h2_ele_EseedOPVsE = nullptr;
  h1_ele_EoPout = nullptr;
  h1_ele_EoPout_barrel = nullptr;
  h1_ele_EoPout_endcaps = nullptr;
  h2_ele_EoPoutVsEta = nullptr;
  h2_ele_EoPoutVsPhi = nullptr;
  h2_ele_EoPoutVsE = nullptr;
  h1_ele_EeleOPout = nullptr;
  h1_ele_EeleOPout_barrel = nullptr;
  h1_ele_EeleOPout_endcaps = nullptr;
  h2_ele_EeleOPoutVsEta = nullptr;
  h2_ele_EeleOPoutVsPhi = nullptr;
  h2_ele_EeleOPoutVsE = nullptr;

  h1_ele_dEtaSc_propVtx = nullptr;
  h1_ele_dEtaSc_propVtx_barrel = nullptr;
  h1_ele_dEtaSc_propVtx_endcaps = nullptr;
  h2_ele_dEtaScVsEta_propVtx = nullptr;
  h2_ele_dEtaScVsPhi_propVtx = nullptr;
  h2_ele_dEtaScVsPt_propVtx = nullptr;
  h1_ele_dPhiSc_propVtx = nullptr;
  h1_ele_dPhiSc_propVtx_barrel = nullptr;
  h1_ele_dPhiSc_propVtx_endcaps = nullptr;
  h2_ele_dPhiScVsEta_propVtx = nullptr;
  h2_ele_dPhiScVsPhi_propVtx = nullptr;
  h2_ele_dPhiScVsPt_propVtx = nullptr;
  h1_ele_dEtaCl_propOut = nullptr;
  h1_ele_dEtaCl_propOut_barrel = nullptr;
  h1_ele_dEtaCl_propOut_endcaps = nullptr;
  h2_ele_dEtaClVsEta_propOut = nullptr;
  h2_ele_dEtaClVsPhi_propOut = nullptr;
  h2_ele_dEtaClVsPt_propOut = nullptr;
  h1_ele_dPhiCl_propOut = nullptr;
  h1_ele_dPhiCl_propOut_barrel = nullptr;
  h1_ele_dPhiCl_propOut_endcaps = nullptr;
  h2_ele_dPhiClVsEta_propOut = nullptr;
  h2_ele_dPhiClVsPhi_propOut = nullptr;
  h2_ele_dPhiClVsPt_propOut = nullptr;
  h1_ele_dEtaEleCl_propOut = nullptr;
  h1_ele_dEtaEleCl_propOut_barrel = nullptr;
  h1_ele_dEtaEleCl_propOut_endcaps = nullptr;
  h2_ele_dEtaEleClVsEta_propOut = nullptr;
  h2_ele_dEtaEleClVsPhi_propOut = nullptr;
  h2_ele_dEtaEleClVsPt_propOut = nullptr;
  h1_ele_dPhiEleCl_propOut = nullptr;
  h1_ele_dPhiEleCl_propOut_barrel = nullptr;
  h1_ele_dPhiEleCl_propOut_endcaps = nullptr;
  h2_ele_dPhiEleClVsEta_propOut = nullptr;
  h2_ele_dPhiEleClVsPhi_propOut = nullptr;
  h2_ele_dPhiEleClVsPt_propOut = nullptr;

  h1_ele_seed_subdet2_ = nullptr;
  h1_ele_seed_mask_ = nullptr;
  h1_ele_seed_mask_bpix_ = nullptr;
  h1_ele_seed_mask_fpix_ = nullptr;
  h1_ele_seed_mask_tec_ = nullptr;
  h1_ele_seed_dphi2_ = nullptr;
  h2_ele_seed_dphi2VsEta_ = nullptr;
  h2_ele_seed_dphi2VsPt_ = nullptr;
  h1_ele_seed_dphi2pos_ = nullptr;
  h2_ele_seed_dphi2posVsEta_ = nullptr;
  h2_ele_seed_dphi2posVsPt_ = nullptr;
  h1_ele_seed_drz2_ = nullptr;
  h2_ele_seed_drz2VsEta_ = nullptr;
  h2_ele_seed_drz2VsPt_ = nullptr;
  h1_ele_seed_drz2pos_ = nullptr;
  h2_ele_seed_drz2posVsEta_ = nullptr;
  h2_ele_seed_drz2posVsPt_ = nullptr;

  h1_ele_classes = nullptr;
  h1_ele_eta = nullptr;
  h1_ele_eta_golden = nullptr;
  h1_ele_eta_bbrem = nullptr;
  h1_ele_eta_narrow = nullptr;
  h1_ele_eta_shower = nullptr;

  h1_ele_HoE = nullptr;
  h1_ele_HoE_barrel = nullptr;
  h1_ele_HoE_endcaps = nullptr;
  h1_ele_HoE_fiducial = nullptr;
  h2_ele_HoEVsEta = nullptr;
  h2_ele_HoEVsPhi = nullptr;
  h2_ele_HoEVsE = nullptr;
  //  h1_scl_ESFrac = 0 ;
  h1_scl_ESFrac_endcaps = nullptr;

  h1_ele_fbrem = nullptr;
  p1_ele_fbremVsEta_mode = nullptr;
  p1_ele_fbremVsEta_mean = nullptr;
  h1_ele_superclusterfbrem = nullptr;
  h1_ele_superclusterfbrem_barrel = nullptr;
  h1_ele_superclusterfbrem_endcaps = nullptr;
  h2_ele_PinVsPoutGolden_mode = nullptr;
  h2_ele_PinVsPoutShowering_mode = nullptr;
  h2_ele_PinVsPoutGolden_mean = nullptr;
  h2_ele_PinVsPoutShowering_mean = nullptr;
  h2_ele_PtinVsPtoutGolden_mode = nullptr;
  h2_ele_PtinVsPtoutShowering_mode = nullptr;
  h2_ele_PtinVsPtoutGolden_mean = nullptr;
  h2_ele_PtinVsPtoutShowering_mean = nullptr;
  h1_scl_EoEmatchingObjectGolden_barrel = nullptr;
  h1_scl_EoEmatchingObjectGolden_endcaps = nullptr;
  h1_scl_EoEmatchingObjectShowering_barrel = nullptr;
  h1_scl_EoEmatchingObjectShowering_endcaps = nullptr;

  h1_ele_mva = nullptr;
  h1_ele_mva_isolated = nullptr;
  h1_ele_provenance = nullptr;

  h1_ele_tkSumPt_dr03 = nullptr;
  h1_ele_tkSumPt_dr03_barrel = nullptr;
  h1_ele_tkSumPt_dr03_endcaps = nullptr;
  h1_ele_ecalRecHitSumEt_dr03 = nullptr;
  h1_ele_ecalRecHitSumEt_dr03_barrel = nullptr;
  h1_ele_ecalRecHitSumEt_dr03_endcaps = nullptr;
  h1_ele_hcalTowerSumEt_dr03_depth1 = nullptr;
  h1_ele_hcalTowerSumEt_dr03_depth1_barrel = nullptr;
  h1_ele_hcalTowerSumEt_dr03_depth1_endcaps = nullptr;
  h1_ele_hcalTowerSumEt_dr03_depth2 = nullptr;
  h1_ele_tkSumPt_dr04 = nullptr;
  h1_ele_tkSumPt_dr04_barrel = nullptr;
  h1_ele_tkSumPt_dr04_endcaps = nullptr;
  h1_ele_ecalRecHitSumEt_dr04 = nullptr;
  h1_ele_ecalRecHitSumEt_dr04_barrel = nullptr;
  h1_ele_ecalRecHitSumEt_dr04_endcaps = nullptr;
  h1_ele_hcalTowerSumEt_dr04_depth1 = nullptr;
  h1_ele_hcalTowerSumEt_dr04_depth1_barrel = nullptr;
  h1_ele_hcalTowerSumEt_dr04_depth1_endcaps = nullptr;
  h1_ele_hcalTowerSumEt_dr04_depth2 = nullptr;

  h1_ele_convFlags = nullptr;
  h1_ele_convFlags_all = nullptr;
  h1_ele_convDist = nullptr;
  h1_ele_convDist_all = nullptr;
  h1_ele_convDcot = nullptr;
  h1_ele_convDcot_all = nullptr;
  h1_ele_convRadius = nullptr;
  h1_ele_convRadius_all = nullptr;
}

void ElectronMcFakeValidator::bookHistograms(DQMStore::IBooker &iBooker, edm::Run const &, edm::EventSetup const &) {
  iBooker.setCurrentFolder(outputInternalPath_);

  setBookIndex(-1);
  setBookPrefix("h");
  setBookEfficiencyFlag(set_EfficiencyFlag);
  setBookStatOverflowFlag(set_StatOverflowFlag);

  // matching object type
  std::string matchingObjectType;
  // Emilia
  matchingObjectType = "GenJet";

  std::string htitle = "# " + matchingObjectType + "s", xtitle = "N_{" + matchingObjectType + "}";
  h1_matchingObjectNum = bookH1withSumw2(iBooker, "matchingObjectNum", htitle, fhits_nbin, 0., fhits_max, xtitle);

  // rec event collections sizes
  h1_recEleNum_ = bookH1(iBooker, "recEleNum", "# rec electrons", ele_nbin, ele_min, ele_max, "N_{ele}");
  h1_recCoreNum_ = bookH1(iBooker, "recCoreNum", "# rec electron cores", core_nbin, core_min, core_max, "N_{core}");
  h1_recTrackNum_ = bookH1(iBooker, "recTrackNum", "# rec gsf tracks", track_nbin, track_min, track_max, "N_{track}");
  h1_recSeedNum_ = bookH1(iBooker, "recSeedNum", "# rec electron seeds", seed_nbin, seed_min, seed_max, "N_{seed}");
  h1_recOfflineVertices_ = bookH1(
      iBooker, "recOfflineVertices", "# rec Offline Primary Vertices", opv_nbin, opv_min, opv_max, "N_{Vertices}");

  // mc
  h1_matchingObjectEta =
      bookH1withSumw2(iBooker, "matchingObject_eta", matchingObjectType + " #eta", eta_nbin, eta_min, eta_max, "#eta");
  h1_matchingObjectAbsEta =
      bookH1withSumw2(iBooker, "matchingObject_abseta", matchingObjectType + " |#eta|", eta_nbin / 2, 0., eta_max);
  h1_matchingObjectP =
      bookH1withSumw2(iBooker, "matchingObject_P", matchingObjectType + " p", p_nbin, 0., p_max, "p (GeV/c)");
  h1_matchingObjectPt =
      bookH1withSumw2(iBooker, "matchingObject_Pt", matchingObjectType + " pt", pteff_nbin, 5., pt_max);
  h1_matchingObjectPhi =
      bookH1withSumw2(iBooker, "matchingObject_phi", matchingObjectType + " phi", phi_nbin, phi_min, phi_max);
  h1_matchingObjectZ = bookH1withSumw2(iBooker, "matchingObject_z", matchingObjectType + " z", xyz_nbin, -25, 25);

  setBookPrefix("h_ele");

  // all electrons
  h1_ele_EoverP_all = bookH1withSumw2(iBooker,
                                      "EoverP_all",
                                      "ele E/P_{vertex}, all reco electrons",
                                      eop_nbin,
                                      0.,
                                      eop_max,
                                      "E/P_{vertex}",
                                      "Events",
                                      "ELE_LOGY E1 P");
  h1_ele_EseedOP_all = bookH1withSumw2(iBooker,
                                       "EseedOP_all",
                                       "ele E_{seed}/P_{vertex}, all reco electrons",
                                       eop_nbin,
                                       0.,
                                       eop_max,
                                       "E_{seed}/P_{vertex}",
                                       "Events",
                                       "ELE_LOGY E1 P");
  h1_ele_EoPout_all = bookH1withSumw2(iBooker,
                                      "EoPout_all",
                                      "ele E_{seed}/P_{out}, all reco electrons",
                                      eop_nbin,
                                      0.,
                                      eop_max,
                                      "E_{seed}/P_{out}",
                                      "Events",
                                      "ELE_LOGY E1 P");
  h1_ele_EeleOPout_all = bookH1withSumw2(iBooker,
                                         "EeleOPout_all",
                                         "ele E_{ele}/P_{out}, all reco electrons",
                                         eop_nbin,
                                         0.,
                                         eop_max,
                                         "E_{ele}/P_{out}",
                                         "Events",
                                         "ELE_LOGY E1 P");
  h1_ele_dEtaSc_propVtx_all = bookH1withSumw2(iBooker,
                                              "dEtaSc_propVtx_all",
                                              "ele #eta_{sc} - #eta_{tr}, prop from vertex, all reco electrons",
                                              detamatch_nbin,
                                              detamatch_min,
                                              detamatch_max,
                                              "#eta_{sc} - #eta_{tr}",
                                              "Events",
                                              "ELE_LOGY E1 P");
  h1_ele_dPhiSc_propVtx_all = bookH1withSumw2(iBooker,
                                              "dPhiSc_propVtx_all",
                                              "ele #phi_{sc} - #phi_{tr}, prop from vertex, all reco electrons",
                                              dphimatch_nbin,
                                              dphimatch_min,
                                              dphimatch_max,
                                              "#phi_{sc} - #phi_{tr} (rad)",
                                              "Events",
                                              "ELE_LOGY E1 P");
  h1_ele_dEtaCl_propOut_all = bookH1withSumw2(iBooker,
                                              "dEtaCl_propOut_all",
                                              "ele #eta_{cl} - #eta_{tr}, prop from outermost, all reco electrons",
                                              detamatch_nbin,
                                              detamatch_min,
                                              detamatch_max,
                                              "#eta_{sc} - #eta_{tr}",
                                              "Events",
                                              "ELE_LOGY E1 P");
  h1_ele_dPhiCl_propOut_all = bookH1withSumw2(iBooker,
                                              "dPhiCl_propOut_all",
                                              "ele #phi_{cl} - #phi_{tr}, prop from outermost, all reco electrons",
                                              dphimatch_nbin,
                                              dphimatch_min,
                                              dphimatch_max,
                                              "#phi_{sc} - #phi_{tr} (rad)",
                                              "Events",
                                              "ELE_LOGY E1 P");
  h1_ele_TIP_all = bookH1withSumw2(iBooker,
                                   "TIP_all",
                                   "ele vertex transverse radius, all reco electrons",
                                   100,
                                   0.,
                                   0.2,
                                   "r_{T} (cm)",
                                   "Events",
                                   "ELE_LOGY E1 P");
  h1_ele_HoE_all = bookH1withSumw2(iBooker,
                                   "HoE_all",
                                   "ele hadronic energy / em energy, all reco electrons",
                                   hoe_nbin,
                                   hoe_min,
                                   hoe_max,
                                   "H/E",
                                   "Events",
                                   "ELE_LOGY E1 P");
  h1_ele_HoE_bc_all = bookH1withSumw2(iBooker,
                                      "HoE_bc_all",
                                      "ele hadronic energy / em energy, all reco electrons, behind cluster",
                                      hoe_nbin,
                                      hoe_min,
                                      hoe_max,
                                      "H/E",
                                      "Events",
                                      "ELE_LOGY E1 P");
  h1_ele_vertexEta_all = bookH1withSumw2(iBooker,
                                         "vertexEta_all",
                                         "ele eta, all reco electrons",
                                         eta_nbin,
                                         eta_min,
                                         eta_max,
                                         "",
                                         "Events",
                                         "ELE_LOGY E1 P");
  h1_ele_vertexPt_all = bookH1withSumw2(
      iBooker, "vertexPt_all", "ele p_{T}, all reco electrons", pteff_nbin, 5., pt_max, "", "Events", "ELE_LOGY E1 P");
  h1_ele_mee_all = bookH1withSumw2(iBooker,
                                   "mee_all",
                                   "ele pairs invariant mass, all reco electrons",
                                   mee_nbin,
                                   mee_min,
                                   mee_max,
                                   "m_{ee} (GeV/c^{2})");
  h1_ele_mee_os = bookH1withSumw2(iBooker,
                                  "mee_os",
                                  "ele pairs invariant mass, opp. sign",
                                  mee_nbin,
                                  mee_min,
                                  mee_max,
                                  "m_{e^{+}e^{-}} (GeV/c^{2})");

  // duplicates
  h2_ele_E2mnE1vsMee_all = bookH2(iBooker,
                                  "E2mnE1vsMee_all",
                                  "E2 - E1 vs ele pairs invariant mass, all electrons",
                                  mee_nbin,
                                  mee_min,
                                  mee_max,
                                  100,
                                  -50.,
                                  50.,
                                  "m_{e^{+}e^{-}} (GeV/c^{2})",
                                  "E2 - E1 (GeV)");
  h2_ele_E2mnE1vsMee_egeg_all = bookH2(iBooker,
                                       "E2mnE1vsMee_egeg_all",
                                       "E2 - E1 vs ele pairs invariant mass, ecal driven pairs, all electrons",
                                       mee_nbin,
                                       mee_min,
                                       mee_max,
                                       100,
                                       -50.,
                                       50.,
                                       "m_{e^{+}e^{-}} (GeV/c^{2})",
                                       "E2 - E1 (GeV)");

  // matched electrons

  htitle = "Efficiency vs matching " + matchingObjectType + " ";
  h1_ele_matchingObjectEta_matched =
      bookH1withSumw2(iBooker, "matchingObjectEta_matched", htitle + "#eta", eta_nbin, eta_min, eta_max);
  h1_ele_matchingObjectAbsEta_matched =
      bookH1withSumw2(iBooker, "matchingObjectAbsEta_matched", htitle + "|#eta|", eta_nbin / 2, 0., eta_max);
  h1_ele_matchingObjectPt_matched =
      bookH1(iBooker, "matchingObjectPt_matched", htitle + "p_{T}", pteff_nbin, 5., pt_max);
  h1_ele_matchingObjectPhi_matched =
      bookH1withSumw2(iBooker, "matchingObjectPhi_matched", htitle + "phi", phi_nbin, phi_min, phi_max);
  h1_ele_matchingObjectZ_matched = bookH1withSumw2(iBooker, "matchingObjectZ_matched", htitle + "z", xyz_nbin, -25, 25);

  h1_ele_charge = bookH1withSumw2(iBooker, "charge", "ele charge", 5, -2.5, 2.5, "charge");
  h2_ele_chargeVsEta = bookH2(iBooker, "chargeVsEta", "ele charge vs eta", eta2D_nbin, eta_min, eta_max, 5, -2., 2.);
  h2_ele_chargeVsPhi = bookH2(iBooker, "chargeVsPhi", "ele charge vs phi", phi2D_nbin, phi_min, phi_max, 5, -2., 2.);
  h2_ele_chargeVsPt = bookH2(iBooker, "chargeVsPt", "ele charge vs pt", pt_nbin, 0., 100., 5, -2., 2.);
  h1_ele_vertexP = bookH1withSumw2(iBooker, "vertexP", "ele momentum", p_nbin, 0., p_max, "p_{vertex} (GeV/c)");
  h1_ele_vertexPt =
      bookH1withSumw2(iBooker, "vertexPt", "ele transverse momentum", pt_nbin, 0., pt_max, "p_{T vertex} (GeV/c)");
  h2_ele_vertexPtVsEta = bookH2(
      iBooker, "vertexPtVsEta", "ele transverse momentum vs eta", eta2D_nbin, eta_min, eta_max, pt2D_nbin, 0., pt_max);
  h2_ele_vertexPtVsPhi = bookH2(
      iBooker, "vertexPtVsPhi", "ele transverse momentum vs phi", phi2D_nbin, phi_min, phi_max, pt2D_nbin, 0., pt_max);
  h1_ele_vertexEta = bookH1withSumw2(iBooker, "vertexEta", "ele momentum eta", eta_nbin, eta_min, eta_max, "#eta");
  h2_ele_vertexEtaVsPhi = bookH2(
      iBooker, "vertexEtaVsPhi", "ele momentum eta vs phi", eta2D_nbin, eta_min, eta_max, phi2D_nbin, phi_min, phi_max);
  h1_ele_vertexPhi =
      bookH1withSumw2(iBooker, "vertexPhi", "ele  momentum #phi", phi_nbin, phi_min, phi_max, "#phi (rad)");
  h1_ele_vertexX = bookH1withSumw2(iBooker, "vertexX", "ele vertex x", xyz_nbin, -0.6, 0.6, "x (cm)");
  h1_ele_vertexY = bookH1withSumw2(iBooker, "vertexY", "ele vertex y", xyz_nbin, -0.6, 0.6, "y (cm)");
  h1_ele_vertexZ = bookH1withSumw2(iBooker, "vertexZ", "ele vertex z", xyz_nbin, -25, 25, "z (cm)");
  h1_ele_vertexTIP = bookH1withSumw2(iBooker,
                                     "vertexTIP",
                                     "ele transverse impact parameter (wrt gen vtx)",
                                     90,
                                     0.,
                                     0.15,
                                     "TIP (cm)",
                                     "Events",
                                     "ELE_LOGY E1 P");
  h2_ele_vertexTIPVsEta = bookH2(iBooker,
                                 "vertexTIPVsEta",
                                 "ele transverse impact parameter (wrt gen vtx) vs eta",
                                 eta2D_nbin,
                                 eta_min,
                                 eta_max,
                                 45,
                                 0.,
                                 0.15,
                                 "#eta",
                                 "TIP (cm)");
  h2_ele_vertexTIPVsPhi = bookH2(iBooker,
                                 "vertexTIPVsPhi",
                                 "ele transverse impact parameter (wrt gen vtx) vs phi",
                                 phi2D_nbin,
                                 phi_min,
                                 phi_max,
                                 45,
                                 0.,
                                 0.15,
                                 "#phi (rad)",
                                 "TIP (cm)");
  h2_ele_vertexTIPVsPt = bookH2(iBooker,
                                "vertexTIPVsPt",
                                "ele transverse impact parameter (wrt gen vtx) vs transverse momentum",
                                pt2D_nbin,
                                0.,
                                pt_max,
                                45,
                                0.,
                                0.15,
                                "p_{T} (GeV/c)",
                                "TIP (cm)");

  htitle = "Electron / Matching " + matchingObjectType + ", momemtum";
  xtitle = "P / P_{" + matchingObjectType + "}";
  h1_ele_PoPmatchingObject =
      bookH1withSumw2(iBooker, "PoPmatchingObject", htitle, popmatching_nbin, popmatching_min, popmatching_max, xtitle);
  h2_ele_PoPmatchingObjectVsEta = bookH2(iBooker,
                                         "PoPmatchingObjectVsEta",
                                         htitle + ",vs eta",
                                         eta2D_nbin,
                                         eta_min,
                                         eta_max,
                                         50,
                                         popmatching_min,
                                         popmatching_max);
  h2_ele_PoPmatchingObjectVsPhi = bookH2(iBooker,
                                         "PoPmatchingObjectVsPhi",
                                         htitle + ",vs phi",
                                         phi2D_nbin,
                                         phi_min,
                                         phi_max,
                                         50,
                                         popmatching_min,
                                         popmatching_max);
  h2_ele_PoPmatchingObjectVsPt = bookH2(
      iBooker, "PoPmatchingObjectVsPt", htitle + ",vs eta", pt2D_nbin, 0., pt_max, 50, popmatching_min, popmatching_max);
  h1_ele_PoPmatchingObject_barrel = bookH1withSumw2(iBooker,
                                                    "PoPmatchingObject_barrel",
                                                    htitle + ", barrel",
                                                    popmatching_nbin,
                                                    popmatching_min,
                                                    popmatching_max,
                                                    xtitle);
  h1_ele_PoPmatchingObject_endcaps = bookH1withSumw2(iBooker,
                                                     "PoPmatchingObject_endcaps",
                                                     htitle + ", endcaps",
                                                     popmatching_nbin,
                                                     popmatching_min,
                                                     popmatching_max,
                                                     xtitle);
  htitle = "Ele - " + matchingObjectType + ", ";
  xtitle = "#eta - #eta_{" + matchingObjectType + "}";
  h1_ele_EtaMnEtamatchingObject =
      bookH1withSumw2(iBooker, "EtamatchingObjectEtaTrue", htitle + "eta", deta_nbin, deta_min, deta_max, xtitle);
  h2_ele_EtaMnEtamatchingObjectVsEta = bookH2(iBooker,
                                              "EtaMnEtamatchingObjectVsEta",
                                              htitle + "eta, vs eta",
                                              eta2D_nbin,
                                              eta_min,
                                              eta_max,
                                              deta_nbin / 2,
                                              deta_min,
                                              deta_max);
  h2_ele_EtaMnEtamatchingObjectVsPhi = bookH2(iBooker,
                                              "EtaMnEtamatchingObjectVsPhi",
                                              htitle + "eta, vs phi",
                                              phi2D_nbin,
                                              phi_min,
                                              phi_max,
                                              deta_nbin / 2,
                                              deta_min,
                                              deta_max);
  h2_ele_EtaMnEtamatchingObjectVsPt = bookH2(iBooker,
                                             "EtaMnEtamatchingObjectVsPt",
                                             htitle + "eta,, vs pt",
                                             pt_nbin,
                                             0.,
                                             pt_max,
                                             deta_nbin / 2,
                                             deta_min,
                                             deta_max);
  xtitle = "#phi - #phi_{" + matchingObjectType + "} (rad)";
  h1_ele_PhiMnPhimatchingObject =
      bookH1withSumw2(iBooker, "PhiMnPhimatchingObject", htitle + "phi", dphi_nbin, dphi_min, dphi_max, xtitle);
  h1_ele_PhiMnPhimatchingObject2 =
      bookH1(iBooker, "PhiMnPhimatchingObject2", htitle + "phi", dphimatch2D_nbin, dphimatch_min, dphimatch_max);
  h2_ele_PhiMnPhimatchingObjectVsEta = bookH2(iBooker,
                                              "PhiMnPhimatchingObjectVsEta",
                                              htitle + "phi, vs eta",
                                              eta2D_nbin,
                                              eta_min,
                                              eta_max,
                                              dphi_nbin / 2,
                                              dphi_min,
                                              dphi_max);
  h2_ele_PhiMnPhimatchingObjectVsPhi = bookH2(iBooker,
                                              "PhiMnPhimatchingObjectVsPhi",
                                              htitle + "phi, vs phi",
                                              phi2D_nbin,
                                              phi_min,
                                              phi_max,
                                              dphi_nbin / 2,
                                              dphi_min,
                                              dphi_max);
  h2_ele_PhiMnPhimatchingObjectVsPt = bookH2(iBooker,
                                             "PhiMnPhimatchingObjectVsPt",
                                             htitle + "phi, vs pt",
                                             pt2D_nbin,
                                             0.,
                                             pt_max,
                                             dphi_nbin / 2,
                                             dphi_min,
                                             dphi_max);

  // matched electron, superclusters

  setBookPrefix("h_scl");

  h1_scl_En_ = bookH1withSumw2(iBooker, "energy", "ele supercluster energy", p_nbin, 0., p_max);
  htitle = "Ele supercluster / " + matchingObjectType + ", energy";
  xtitle = "E/E_{" + matchingObjectType + "}";
  h1_scl_EoEmatchingObject_barrel =
      bookH1withSumw2(iBooker, "EoEmatchingObject_barrel", htitle + ", barrel", 50, 0.2, 1.2, xtitle);
  h1_scl_EoEmatchingObject_endcaps =
      bookH1withSumw2(iBooker, "EoEmatchingObject_endcaps", htitle + ", endcaps", 50, 0.2, 1.2, xtitle);
  h1_scl_Et_ = bookH1withSumw2(iBooker, "et", "ele supercluster transverse energy", pt_nbin, 0., pt_max);
  h2_scl_EtVsEta_ = bookH2(iBooker,
                           "etVsEta",
                           "ele supercluster transverse energy vs eta",
                           eta2D_nbin,
                           eta_min,
                           eta_max,
                           pt_nbin,
                           0.,
                           pt_max);
  h2_scl_EtVsPhi_ = bookH2(iBooker,
                           "etVsPhi",
                           "ele supercluster transverse energy vs phi",
                           phi2D_nbin,
                           phi_min,
                           phi_max,
                           pt_nbin,
                           0.,
                           pt_max);
  h2_scl_EtaVsPhi_ = bookH2(
      iBooker, "etaVsPhi", "ele supercluster eta vs phi", phi2D_nbin, phi_min, phi_max, eta2D_nbin, eta_min, eta_max);
  h1_scl_Eta_ = bookH1withSumw2(iBooker, "eta", "ele supercluster eta", eta_nbin, eta_min, eta_max);
  h1_scl_Phi_ = bookH1withSumw2(iBooker, "phi", "ele supercluster phi", phi_nbin, phi_min, phi_max);
  h1_scl_SigIEtaIEta_ = bookH1withSumw2(iBooker,
                                        "sigietaieta",
                                        "ele supercluster sigma ieta ieta",
                                        100,
                                        0.,
                                        0.05,
                                        "#sigma_{i#eta i#eta}",
                                        "Events",
                                        "ELE_LOGY E1 P");
  h1_scl_SigIEtaIEta_barrel_ = bookH1withSumw2(iBooker,
                                               "sigietaieta_barrel",
                                               "ele supercluster sigma ieta ieta, barrel",
                                               100,
                                               0.,
                                               0.05,
                                               "#sigma_{i#eta i#eta}",
                                               "Events",
                                               "ELE_LOGY E1 P");
  h1_scl_SigIEtaIEta_endcaps_ = bookH1withSumw2(iBooker,
                                                "sigietaieta_endcaps",
                                                "ele supercluster sigma ieta ieta, endcaps",
                                                100,
                                                0.,
                                                0.05,
                                                "#sigma_{i#eta i#eta}",
                                                "Events",
                                                "ELE_LOGY E1 P");
  // new 2014.01.12
  h1_scl_full5x5_sigmaIetaIeta_ = bookH1withSumw2(iBooker,
                                                  "full5x5_sigietaieta",
                                                  "ele supercluster full5x5 sigma ieta ieta",
                                                  100,
                                                  0.,
                                                  0.05,
                                                  "#sigma_{i#eta i#eta}",
                                                  "Events",
                                                  "ELE_LOGY E1 P");
  h1_scl_full5x5_sigmaIetaIeta_barrel_ = bookH1withSumw2(iBooker,
                                                         "full5x5_sigietaieta_barrel",
                                                         "ele supercluster full5x5 sigma ieta ieta, barrel",
                                                         100,
                                                         0.,
                                                         0.05,
                                                         "#sigma_{i#eta i#eta}",
                                                         "Events",
                                                         "ELE_LOGY E1 P");
  h1_scl_full5x5_sigmaIetaIeta_endcaps_ = bookH1withSumw2(iBooker,
                                                          "full5x5_sigietaieta_endcaps",
                                                          "ele supercluster full5x5 sigma ieta ieta, endcaps",
                                                          100,
                                                          0.,
                                                          0.05,
                                                          "#sigma_{i#eta i#eta}",
                                                          "Events",
                                                          "ELE_LOGY E1 P");
  // new 2014.01.12
  h1_scl_E1x5_ = bookH1withSumw2(
      iBooker, "E1x5", "ele supercluster energy in 1x5", p_nbin, 0., p_max, "E1x5 (GeV)", "Events", "ELE_LOGY E1 P");
  h1_scl_E1x5_barrel_ = bookH1withSumw2(iBooker,
                                        "E1x5_barrel",
                                        "ele supercluster energy in 1x5 barrel",
                                        p_nbin,
                                        0.,
                                        p_max,
                                        "E1x5 (GeV)",
                                        "Events",
                                        "ELE_LOGY E1 P");
  h1_scl_E1x5_endcaps_ = bookH1withSumw2(iBooker,
                                         "E1x5_endcaps",
                                         "ele supercluster energy in 1x5 endcaps",
                                         p_nbin,
                                         0.,
                                         p_max,
                                         "E1x5 (GeV)",
                                         "Events",
                                         "ELE_LOGY E1 P");
  h1_scl_E2x5max_ = bookH1withSumw2(iBooker,
                                    "E2x5max",
                                    "ele supercluster energy in 2x5 max",
                                    p_nbin,
                                    0.,
                                    p_max,
                                    "E2x5 (GeV)",
                                    "Events",
                                    "ELE_LOGY E1 P");
  h1_scl_E2x5max_barrel_ = bookH1withSumw2(iBooker,
                                           "E2x5max_barrel",
                                           "ele supercluster energy in 2x5 _max barrel",
                                           p_nbin,
                                           0.,
                                           p_max,
                                           "E2x5 (GeV)",
                                           "Events",
                                           "ELE_LOGY E1 P");
  h1_scl_E2x5max_endcaps_ = bookH1withSumw2(iBooker,
                                            "E2x5max_endcaps",
                                            "ele supercluster energy in 2x5 _max endcaps",
                                            p_nbin,
                                            0.,
                                            p_max,
                                            "E2x5 (GeV)",
                                            "Events",
                                            "ELE_LOGY E1 P");
  h1_scl_E5x5_ = bookH1withSumw2(
      iBooker, "E5x5", "ele supercluster energy in 5x5", p_nbin, 0., p_max, "E5x5 (GeV)", "Events", "ELE_LOGY E1 P");
  h1_scl_E5x5_barrel_ = bookH1withSumw2(iBooker,
                                        "E5x5_barrel",
                                        "ele supercluster energy in 5x5 barrel",
                                        p_nbin,
                                        0.,
                                        p_max,
                                        "E5x5 (GeV)",
                                        "Events",
                                        "ELE_LOGY E1 P");
  h1_scl_E5x5_endcaps_ = bookH1withSumw2(iBooker,
                                         "E5x5_endcaps",
                                         "ele supercluster energy in 5x5 endcaps",
                                         p_nbin,
                                         0.,
                                         p_max,
                                         "E5x5 (GeV)",
                                         "Events",
                                         "ELE_LOGY E1 P");

  // matched electron, gsf tracks

  setBookPrefix("h_ele");

  h1_ele_ambiguousTracks = bookH1withSumw2(iBooker,
                                           "ambiguousTracks",
                                           "ele # ambiguous tracks",
                                           5,
                                           0.,
                                           5.,
                                           "N_{ambiguous tracks}",
                                           "Events",
                                           "ELE_LOGY E1 P");
  h2_ele_ambiguousTracksVsEta = bookH2(
      iBooker, "ambiguousTracksVsEta", "ele # ambiguous tracks  vs eta", eta2D_nbin, eta_min, eta_max, 5, 0., 5.);
  h2_ele_ambiguousTracksVsPhi = bookH2(
      iBooker, "ambiguousTracksVsPhi", "ele # ambiguous tracks  vs phi", phi2D_nbin, phi_min, phi_max, 5, 0., 5.);
  h2_ele_ambiguousTracksVsPt =
      bookH2(iBooker, "ambiguousTracksVsPt", "ele # ambiguous tracks vs pt", pt2D_nbin, 0., pt_max, 5, 0., 5.);
  h1_ele_foundHits =
      bookH1withSumw2(iBooker, "foundHits", "ele track # found hits", fhits_nbin, 0., fhits_max, "N_{hits}");
  h2_ele_foundHitsVsEta = bookH2(iBooker,
                                 "foundHitsVsEta",
                                 "ele track # found hits vs eta",
                                 eta2D_nbin,
                                 eta_min,
                                 eta_max,
                                 fhits_nbin,
                                 0.,
                                 fhits_max);
  h2_ele_foundHitsVsPhi = bookH2(iBooker,
                                 "foundHitsVsPhi",
                                 "ele track # found hits vs phi",
                                 phi2D_nbin,
                                 phi_min,
                                 phi_max,
                                 fhits_nbin,
                                 0.,
                                 fhits_max);
  h2_ele_foundHitsVsPt = bookH2(
      iBooker, "foundHitsVsPt", "ele track # found hits vs pt", pt2D_nbin, 0., pt_max, fhits_nbin, 0., fhits_max);
  h1_ele_lostHits = bookH1withSumw2(iBooker, "lostHits", "ele track # lost hits", 5, 0., 5., "N_{lost hits}");
  h2_ele_lostHitsVsEta = bookH2(
      iBooker, "lostHitsVsEta", "ele track # lost hits vs eta", eta2D_nbin, eta_min, eta_max, lhits_nbin, 0., lhits_max);
  h2_ele_lostHitsVsPhi = bookH2(
      iBooker, "lostHitsVsPhi", "ele track # lost hits vs eta", phi2D_nbin, phi_min, phi_max, lhits_nbin, 0., lhits_max);
  h2_ele_lostHitsVsPt =
      bookH2(iBooker, "lostHitsVsPt", "ele track # lost hits vs eta", pt2D_nbin, 0., pt_max, lhits_nbin, 0., lhits_max);
  h1_ele_chi2 =
      bookH1withSumw2(iBooker, "chi2", "ele track #chi^{2}", 100, 0., 15., "#Chi^{2}", "Events", "ELE_LOGY E1 P");
  h2_ele_chi2VsEta =
      bookH2(iBooker, "chi2VsEta", "ele track #chi^{2} vs eta", eta2D_nbin, eta_min, eta_max, 50, 0., 15.);
  h2_ele_chi2VsPhi =
      bookH2(iBooker, "chi2VsPhi", "ele track #chi^{2} vs phi", phi2D_nbin, phi_min, phi_max, 50, 0., 15.);
  h2_ele_chi2VsPt = bookH2(iBooker, "chi2VsPt", "ele track #chi^{2} vs pt", pt2D_nbin, 0., pt_max, 50, 0., 15.);
  h1_ele_PinMnPout = bookH1withSumw2(iBooker,
                                     "PinMnPout",
                                     "ele track inner p - outer p, mean of GSF components",
                                     p_nbin,
                                     0.,
                                     200.,
                                     "P_{vertex} - P_{out} (GeV/c)");
  h1_ele_PinMnPout_mode = bookH1withSumw2(iBooker,
                                          "PinMnPout_mode",
                                          "ele track inner p - outer p, mode of GSF components",
                                          p_nbin,
                                          0.,
                                          100.,
                                          "P_{vertex} - P_{out}, mode of GSF components (GeV/c)");
  h2_ele_PinMnPoutVsEta_mode = bookH2(iBooker,
                                      "PinMnPoutVsEta_mode",
                                      "ele track inner p - outer p vs eta, mode of GSF components",
                                      eta2D_nbin,
                                      eta_min,
                                      eta_max,
                                      p2D_nbin,
                                      0.,
                                      100.);
  h2_ele_PinMnPoutVsPhi_mode = bookH2(iBooker,
                                      "PinMnPoutVsPhi_mode",
                                      "ele track inner p - outer p vs phi, mode of GSF components",
                                      phi2D_nbin,
                                      phi_min,
                                      phi_max,
                                      p2D_nbin,
                                      0.,
                                      100.);
  h2_ele_PinMnPoutVsPt_mode = bookH2(iBooker,
                                     "PinMnPoutVsPt_mode",
                                     "ele track inner p - outer p vs pt, mode of GSF components",
                                     pt2D_nbin,
                                     0.,
                                     pt_max,
                                     p2D_nbin,
                                     0.,
                                     100.);
  h2_ele_PinMnPoutVsE_mode = bookH2(iBooker,
                                    "PinMnPoutVsE_mode",
                                    "ele track inner p - outer p vs E, mode of GSF components",
                                    p2D_nbin,
                                    0.,
                                    200.,
                                    p2D_nbin,
                                    0.,
                                    100.);
  h2_ele_PinMnPoutVsChi2_mode = bookH2(iBooker,
                                       "PinMnPoutVsChi2_mode",
                                       "ele track inner p - outer p vs track chi2, mode of GSF components",
                                       50,
                                       0.,
                                       20.,
                                       p2D_nbin,
                                       0.,
                                       100.);
  h1_ele_outerP = bookH1withSumw2(
      iBooker, "outerP", "ele track outer p, mean of GSF components", p_nbin, 0., p_max, "P_{out} (GeV/c)");
  h1_ele_outerP_mode = bookH1withSumw2(
      iBooker, "outerP_mode", "ele track outer p, mode of GSF components", p_nbin, 0., p_max, "P_{out} (GeV/c)");
  h2_ele_outerPVsEta_mode =
      bookH2(iBooker, "outerPVsEta_mode", "ele track outer p vs eta mode", eta2D_nbin, eta_min, eta_max, 50, 0., p_max);
  h1_ele_outerPt = bookH1withSumw2(
      iBooker, "outerPt", "ele track outer p_{T}, mean of GSF components", pt_nbin, 0., pt_max, "P_{T out} (GeV/c)");
  h1_ele_outerPt_mode = bookH1withSumw2(iBooker,
                                        "outerPt_mode",
                                        "ele track outer p_{T}, mode of GSF components",
                                        pt_nbin,
                                        0.,
                                        pt_max,
                                        "P_{T out} (GeV/c)");
  h2_ele_outerPtVsEta_mode = bookH2(iBooker,
                                    "outerPtVsEta_mode",
                                    "ele track outer p_{T} vs eta, mode of GSF components",
                                    eta2D_nbin,
                                    eta_min,
                                    eta_max,
                                    pt2D_nbin,
                                    0.,
                                    pt_max);
  h2_ele_outerPtVsPhi_mode = bookH2(iBooker,
                                    "outerPtVsPhi_mode",
                                    "ele track outer p_{T} vs phi, mode of GSF components",
                                    phi2D_nbin,
                                    phi_min,
                                    phi_max,
                                    pt2D_nbin,
                                    0.,
                                    pt_max);
  h2_ele_outerPtVsPt_mode = bookH2(iBooker,
                                   "outerPtVsPt_mode",
                                   "ele track outer p_{T} vs pt, mode of GSF components",
                                   pt2D_nbin,
                                   0.,
                                   100.,
                                   pt2D_nbin,
                                   0.,
                                   pt_max);

  // matched electrons, matching
  h1_ele_EoP = bookH1withSumw2(
      iBooker, "EoP", "ele E/P_{vertex}", eop_nbin, 0., eop_max, "E/P_{vertex}", "Events", "ELE_LOGY E1 P");
  h1_ele_EoP_barrel = bookH1withSumw2(iBooker,
                                      "EoP_barrel",
                                      "ele E/P_{vertex} barrel",
                                      eop_nbin,
                                      0.,
                                      eop_max,
                                      "E/P_{vertex}",
                                      "Events",
                                      "ELE_LOGY E1 P");
  h1_ele_EoP_endcaps = bookH1withSumw2(iBooker,
                                       "EoP_endcaps",
                                       "ele E/P_{vertex} endcaps",
                                       eop_nbin,
                                       0.,
                                       eop_max,
                                       "E/P_{vertex}",
                                       "Events",
                                       "ELE_LOGY E1 P");
  h2_ele_EoPVsEta =
      bookH2(iBooker, "EoPVsEta", "ele E/P_{vertex} vs eta", eta2D_nbin, eta_min, eta_max, eop2D_nbin, 0., eopmaxsht);
  h2_ele_EoPVsPhi =
      bookH2(iBooker, "EoPVsPhi", "ele E/P_{vertex} vs phi", phi2D_nbin, phi_min, phi_max, eop2D_nbin, 0., eopmaxsht);
  h2_ele_EoPVsE = bookH2(iBooker, "EoPVsE", "ele E/P_{vertex} vs E", 50, 0., p_max, 50, 0., 5.);
  h1_ele_EseedOP = bookH1withSumw2(iBooker,
                                   "EseedOP",
                                   "ele E_{seed}/P_{vertex}",
                                   eop_nbin,
                                   0.,
                                   eop_max,
                                   "E_{seed}/P_{vertex}",
                                   "Events",
                                   "ELE_LOGY E1 P");
  h1_ele_EseedOP_barrel = bookH1withSumw2(iBooker,
                                          "EseedOP_barrel",
                                          "ele E_{seed}/P_{vertex} barrel",
                                          eop_nbin,
                                          0.,
                                          eop_max,
                                          "E_{seed}/P_{vertex}",
                                          "Events",
                                          "ELE_LOGY E1 P");
  h1_ele_EseedOP_endcaps = bookH1withSumw2(iBooker,
                                           "EseedOP_endcaps",
                                           "ele E_{seed}/P_{vertex} endcaps",
                                           eop_nbin,
                                           0.,
                                           eop_max,
                                           "E_{seed}/P_{vertex}",
                                           "Events",
                                           "ELE_LOGY E1 P");
  h2_ele_EseedOPVsEta = bookH2(iBooker,
                               "EseedOPVsEta",
                               "ele E_{seed}/P_{vertex} vs eta",
                               eta2D_nbin,
                               eta_min,
                               eta_max,
                               eop2D_nbin,
                               0.,
                               eopmaxsht);
  h2_ele_EseedOPVsPhi = bookH2(iBooker,
                               "EseedOPVsPhi",
                               "ele E_{seed}/P_{vertex} vs phi",
                               phi2D_nbin,
                               phi_min,
                               phi_max,
                               eop2D_nbin,
                               0.,
                               eopmaxsht);
  h2_ele_EseedOPVsE = bookH2(iBooker, "EseedOPVsE", "ele E_{seed}/P_{vertex} vs E", 50, 0., p_max, 50, 0., 5.);
  h1_ele_EoPout = bookH1withSumw2(
      iBooker, "EoPout", "ele E_{seed}/P_{out}", eop_nbin, 0., eop_max, "E_{seed}/P_{out}", "Events", "ELE_LOGY E1 P");
  h1_ele_EoPout_barrel = bookH1withSumw2(iBooker,
                                         "EoPout_barrel",
                                         "ele E_{seed}/P_{out} barrel",
                                         eop_nbin,
                                         0.,
                                         eop_max,
                                         "E_{seed}/P_{out}",
                                         "Events",
                                         "ELE_LOGY E1 P");
  h1_ele_EoPout_endcaps = bookH1withSumw2(iBooker,
                                          "EoPout_endcaps",
                                          "ele E_{seed}/P_{out} endcaps",
                                          eop_nbin,
                                          0.,
                                          eop_max,
                                          "E_{seed}/P_{out}",
                                          "Events",
                                          "ELE_LOGY E1 P");
  h2_ele_EoPoutVsEta = bookH2(
      iBooker, "EoPoutVsEta", "ele E_{seed}/P_{out} vs eta", eta2D_nbin, eta_min, eta_max, eop2D_nbin, 0., eopmaxsht);
  h2_ele_EoPoutVsPhi = bookH2(
      iBooker, "EoPoutVsPhi", "ele E_{seed}/P_{out} vs phi", phi2D_nbin, phi_min, phi_max, eop2D_nbin, 0., eopmaxsht);
  h2_ele_EoPoutVsE =
      bookH2(iBooker, "EoPoutVsE", "ele E_{seed}/P_{out} vs E", p2D_nbin, 0., p_max, eop2D_nbin, 0., eopmaxsht);
  h1_ele_EeleOPout = bookH1withSumw2(
      iBooker, "EeleOPout", "ele E_{ele}/P_{out}", eop_nbin, 0., eop_max, "E_{ele}/P_{out}", "Events", "ELE_LOGY E1 P");
  h1_ele_EeleOPout_barrel = bookH1withSumw2(iBooker,
                                            "EeleOPout_barrel",
                                            "ele E_{ele}/P_{out} barrel",
                                            eop_nbin,
                                            0.,
                                            eop_max,
                                            "E_{ele}/P_{out}",
                                            "Events",
                                            "ELE_LOGY E1 P");
  h1_ele_EeleOPout_endcaps = bookH1withSumw2(iBooker,
                                             "EeleOPout_endcaps",
                                             "ele E_{ele}/P_{out} endcaps",
                                             eop_nbin,
                                             0.,
                                             eop_max,
                                             "E_{ele}/P_{out}",
                                             "Events",
                                             "ELE_LOGY E1 P");
  h2_ele_EeleOPoutVsEta = bookH2(
      iBooker, "EeleOPoutVsEta", "ele E_{ele}/P_{out} vs eta", eta2D_nbin, eta_min, eta_max, eop2D_nbin, 0., eopmaxsht);
  h2_ele_EeleOPoutVsPhi = bookH2(
      iBooker, "EeleOPoutVsPhi", "ele E_{ele}/P_{out} vs phi", phi2D_nbin, phi_min, phi_max, eop2D_nbin, 0., eopmaxsht);
  h2_ele_EeleOPoutVsE =
      bookH2(iBooker, "EeleOPoutVsE", "ele E_{ele}/P_{out} vs E", p2D_nbin, 0., p_max, eop2D_nbin, 0., eopmaxsht);
  h1_ele_dEtaSc_propVtx = bookH1withSumw2(iBooker,
                                          "dEtaSc_propVtx",
                                          "ele #eta_{sc} - #eta_{tr}, prop from vertex",
                                          detamatch_nbin,
                                          detamatch_min,
                                          detamatch_max,
                                          "#eta_{sc} - #eta_{tr}",
                                          "Events",
                                          "ELE_LOGY E1 P");
  h1_ele_dEtaSc_propVtx_barrel = bookH1withSumw2(iBooker,
                                                 "dEtaSc_propVtx_barrel",
                                                 "ele #eta_{sc} - #eta_{tr}, prop from vertex, barrel",
                                                 detamatch_nbin,
                                                 detamatch_min,
                                                 detamatch_max,
                                                 "#eta_{sc} - #eta_{tr}",
                                                 "Events",
                                                 "ELE_LOGY E1 P");
  h1_ele_dEtaSc_propVtx_endcaps = bookH1withSumw2(iBooker,
                                                  "dEtaSc_propVtx_endcaps",
                                                  "ele #eta_{sc} - #eta_{tr}, prop from vertex, endcaps",
                                                  detamatch_nbin,
                                                  detamatch_min,
                                                  detamatch_max,
                                                  "#eta_{sc} - #eta_{tr}",
                                                  "Events",
                                                  "ELE_LOGY E1 P");
  h2_ele_dEtaScVsEta_propVtx = bookH2(iBooker,
                                      "dEtaScVsEta_propVtx",
                                      "ele #eta_{sc} - #eta_{tr} vs eta, prop from vertex",
                                      eta2D_nbin,
                                      eta_min,
                                      eta_max,
                                      detamatch2D_nbin,
                                      detamatch_min,
                                      detamatch_max);
  h2_ele_dEtaScVsPhi_propVtx = bookH2(iBooker,
                                      "dEtaScVsPhi_propVtx",
                                      "ele #eta_{sc} - #eta_{tr} vs phi, prop from vertex",
                                      phi2D_nbin,
                                      phi_min,
                                      phi_max,
                                      detamatch2D_nbin,
                                      detamatch_min,
                                      detamatch_max);
  h2_ele_dEtaScVsPt_propVtx = bookH2(iBooker,
                                     "dEtaScVsPt_propVtx",
                                     "ele #eta_{sc} - #eta_{tr} vs pt, prop from vertex",
                                     pt2D_nbin,
                                     0.,
                                     pt_max,
                                     detamatch2D_nbin,
                                     detamatch_min,
                                     detamatch_max);
  h1_ele_dPhiSc_propVtx = bookH1withSumw2(iBooker,
                                          "dPhiSc_propVtx",
                                          "ele #phi_{sc} - #phi_{tr}, prop from vertex",
                                          dphimatch_nbin,
                                          dphimatch_min,
                                          dphimatch_max,
                                          "#phi_{sc} - #phi_{tr} (rad)",
                                          "Events",
                                          "ELE_LOGY E1 P");
  h1_ele_dPhiSc_propVtx_barrel = bookH1withSumw2(iBooker,
                                                 "dPhiSc_propVtx_barrel",
                                                 "ele #phi_{sc} - #phi_{tr}, prop from vertex, barrel",
                                                 dphimatch_nbin,
                                                 dphimatch_min,
                                                 dphimatch_max,
                                                 "#phi_{sc} - #phi_{tr} (rad)",
                                                 "Events",
                                                 "ELE_LOGY E1 P");
  h1_ele_dPhiSc_propVtx_endcaps = bookH1withSumw2(iBooker,
                                                  "dPhiSc_propVtx_endcaps",
                                                  "ele #phi_{sc} - #phi_{tr}, prop from vertex, endcaps",
                                                  dphimatch_nbin,
                                                  dphimatch_min,
                                                  dphimatch_max,
                                                  "#phi_{sc} - #phi_{tr} (rad)",
                                                  "Events",
                                                  "ELE_LOGY E1 P");
  h2_ele_dPhiScVsEta_propVtx = bookH2(iBooker,
                                      "dPhiScVsEta_propVtx",
                                      "ele #phi_{sc} - #phi_{tr} vs eta, prop from vertex",
                                      eta2D_nbin,
                                      eta_min,
                                      eta_max,
                                      dphimatch2D_nbin,
                                      dphimatch_min,
                                      dphimatch_max);
  h2_ele_dPhiScVsPhi_propVtx = bookH2(iBooker,
                                      "dPhiScVsPhi_propVtx",
                                      "ele #phi_{sc} - #phi_{tr} vs phi, prop from vertex",
                                      phi2D_nbin,
                                      phi_min,
                                      phi_max,
                                      dphimatch2D_nbin,
                                      dphimatch_min,
                                      dphimatch_max);
  h2_ele_dPhiScVsPt_propVtx = bookH2(iBooker,
                                     "dPhiScVsPt_propVtx",
                                     "ele #phi_{sc} - #phi_{tr} vs pt, prop from vertex",
                                     pt2D_nbin,
                                     0.,
                                     pt_max,
                                     dphimatch2D_nbin,
                                     dphimatch_min,
                                     dphimatch_max);
  h1_ele_dEtaCl_propOut = bookH1withSumw2(iBooker,
                                          "dEtaCl_propOut",
                                          "ele #eta_{cl} - #eta_{tr}, prop from outermost",
                                          detamatch_nbin,
                                          detamatch_min,
                                          detamatch_max,
                                          "#eta_{seedcl} - #eta_{tr}",
                                          "Events",
                                          "ELE_LOGY E1 P");
  h1_ele_dEtaCl_propOut_barrel = bookH1withSumw2(iBooker,
                                                 "dEtaCl_propOut_barrel",
                                                 "ele #eta_{cl} - #eta_{tr}, prop from outermost, barrel",
                                                 detamatch_nbin,
                                                 detamatch_min,
                                                 detamatch_max,
                                                 "#eta_{seedcl} - #eta_{tr}",
                                                 "Events",
                                                 "ELE_LOGY E1 P");
  h1_ele_dEtaCl_propOut_endcaps = bookH1withSumw2(iBooker,
                                                  "dEtaCl_propOut_endcaps",
                                                  "ele #eta_{cl} - #eta_{tr}, prop from outermost, endcaps",
                                                  detamatch_nbin,
                                                  detamatch_min,
                                                  detamatch_max,
                                                  "#eta_{seedcl} - #eta_{tr}",
                                                  "Events",
                                                  "ELE_LOGY E1 P");
  h2_ele_dEtaClVsEta_propOut = bookH2(iBooker,
                                      "dEtaClVsEta_propOut",
                                      "ele #eta_{cl} - #eta_{tr} vs eta, prop from out",
                                      eta2D_nbin,
                                      eta_min,
                                      eta_max,
                                      detamatch2D_nbin,
                                      detamatch_min,
                                      detamatch_max);
  h2_ele_dEtaClVsPhi_propOut = bookH2(iBooker,
                                      "dEtaClVsPhi_propOut",
                                      "ele #eta_{cl} - #eta_{tr} vs phi, prop from out",
                                      phi2D_nbin,
                                      phi_min,
                                      phi_max,
                                      detamatch2D_nbin,
                                      detamatch_min,
                                      detamatch_max);
  h2_ele_dEtaClVsPt_propOut = bookH2(iBooker,
                                     "dEtaScVsPt_propOut",
                                     "ele #eta_{cl} - #eta_{tr} vs pt, prop from out",
                                     pt2D_nbin,
                                     0.,
                                     pt_max,
                                     detamatch2D_nbin,
                                     detamatch_min,
                                     detamatch_max);
  h1_ele_dPhiCl_propOut = bookH1withSumw2(iBooker,
                                          "dPhiCl_propOut",
                                          "ele #phi_{cl} - #phi_{tr}, prop from outermost",
                                          dphimatch_nbin,
                                          dphimatch_min,
                                          dphimatch_max,
                                          "#phi_{seedcl} - #phi_{tr} (rad)",
                                          "Events",
                                          "ELE_LOGY E1 P");
  h1_ele_dPhiCl_propOut_barrel = bookH1withSumw2(iBooker,
                                                 "dPhiCl_propOut_barrel",
                                                 "ele #phi_{cl} - #phi_{tr}, prop from outermost, barrel",
                                                 dphimatch_nbin,
                                                 dphimatch_min,
                                                 dphimatch_max,
                                                 "#phi_{seedcl} - #phi_{tr} (rad)",
                                                 "Events",
                                                 "ELE_LOGY E1 P");
  h1_ele_dPhiCl_propOut_endcaps = bookH1withSumw2(iBooker,
                                                  "dPhiCl_propOut_endcaps",
                                                  "ele #phi_{cl} - #phi_{tr}, prop from outermost, endcaps",
                                                  dphimatch_nbin,
                                                  dphimatch_min,
                                                  dphimatch_max,
                                                  "#phi_{seedcl} - #phi_{tr} (rad)",
                                                  "Events",
                                                  "ELE_LOGY E1 P");
  h2_ele_dPhiClVsEta_propOut = bookH2(iBooker,
                                      "dPhiClVsEta_propOut",
                                      "ele #phi_{cl} - #phi_{tr} vs eta, prop from out",
                                      eta2D_nbin,
                                      eta_min,
                                      eta_max,
                                      dphimatch2D_nbin,
                                      dphimatch_min,
                                      dphimatch_max);
  h2_ele_dPhiClVsPhi_propOut = bookH2(iBooker,
                                      "dPhiClVsPhi_propOut",
                                      "ele #phi_{cl} - #phi_{tr} vs phi, prop from out",
                                      phi2D_nbin,
                                      phi_min,
                                      phi_max,
                                      dphimatch2D_nbin,
                                      dphimatch_min,
                                      dphimatch_max);
  h2_ele_dPhiClVsPt_propOut = bookH2(iBooker,
                                     "dPhiSClsPt_propOut",
                                     "ele #phi_{cl} - #phi_{tr} vs pt, prop from out",
                                     pt2D_nbin,
                                     0.,
                                     pt_max,
                                     dphimatch2D_nbin,
                                     dphimatch_min,
                                     dphimatch_max);
  h1_ele_dEtaEleCl_propOut = bookH1withSumw2(iBooker,
                                             "dEtaEleCl_propOut",
                                             "ele #eta_{EleCl} - #eta_{tr}, prop from outermost",
                                             detamatch_nbin,
                                             detamatch_min,
                                             detamatch_max,
                                             "#eta_{elecl} - #eta_{tr}",
                                             "Events",
                                             "ELE_LOGY E1 P");
  h1_ele_dEtaEleCl_propOut_barrel = bookH1withSumw2(iBooker,
                                                    "dEtaEleCl_propOut_barrel",
                                                    "ele #eta_{EleCl} - #eta_{tr}, prop from outermost, barrel",
                                                    detamatch_nbin,
                                                    detamatch_min,
                                                    detamatch_max,
                                                    "#eta_{elecl} - #eta_{tr}",
                                                    "Events",
                                                    "ELE_LOGY E1 P");
  h1_ele_dEtaEleCl_propOut_endcaps = bookH1withSumw2(iBooker,
                                                     "dEtaEleCl_propOut_endcaps",
                                                     "ele #eta_{EleCl} - #eta_{tr}, prop from outermost, endcaps",
                                                     detamatch_nbin,
                                                     detamatch_min,
                                                     detamatch_max,
                                                     "#eta_{elecl} - #eta_{tr}",
                                                     "Events",
                                                     "ELE_LOGY E1 P");
  h2_ele_dEtaEleClVsEta_propOut = bookH2(iBooker,
                                         "dEtaEleClVsEta_propOut",
                                         "ele #eta_{EleCl} - #eta_{tr} vs eta, prop from out",
                                         eta2D_nbin,
                                         eta_min,
                                         eta_max,
                                         detamatch2D_nbin,
                                         detamatch_min,
                                         detamatch_max);
  h2_ele_dEtaEleClVsPhi_propOut = bookH2(iBooker,
                                         "dEtaEleClVsPhi_propOut",
                                         "ele #eta_{EleCl} - #eta_{tr} vs phi, prop from out",
                                         phi2D_nbin,
                                         phi_min,
                                         phi_max,
                                         detamatch2D_nbin,
                                         detamatch_min,
                                         detamatch_max);
  h2_ele_dEtaEleClVsPt_propOut = bookH2(iBooker,
                                        "dEtaScVsPt_propOut",
                                        "ele #eta_{EleCl} - #eta_{tr} vs pt, prop from out",
                                        pt2D_nbin,
                                        0.,
                                        pt_max,
                                        detamatch2D_nbin,
                                        detamatch_min,
                                        detamatch_max);
  h1_ele_dPhiEleCl_propOut = bookH1withSumw2(iBooker,
                                             "dPhiEleCl_propOut",
                                             "ele #phi_{EleCl} - #phi_{tr}, prop from outermost",
                                             dphimatch_nbin,
                                             dphimatch_min,
                                             dphimatch_max,
                                             "#phi_{elecl} - #phi_{tr} (rad)",
                                             "Events",
                                             "ELE_LOGY E1 P");
  h1_ele_dPhiEleCl_propOut_barrel = bookH1withSumw2(iBooker,
                                                    "dPhiEleCl_propOut_barrel",
                                                    "ele #phi_{EleCl} - #phi_{tr}, prop from outermost, barrel",
                                                    dphimatch_nbin,
                                                    dphimatch_min,
                                                    dphimatch_max,
                                                    "#phi_{elecl} - #phi_{tr} (rad)",
                                                    "Events",
                                                    "ELE_LOGY E1 P");
  h1_ele_dPhiEleCl_propOut_endcaps = bookH1withSumw2(iBooker,
                                                     "dPhiEleCl_propOut_endcaps",
                                                     "ele #phi_{EleCl} - #phi_{tr}, prop from outermost, endcaps",
                                                     dphimatch_nbin,
                                                     dphimatch_min,
                                                     dphimatch_max,
                                                     "#phi_{elecl} - #phi_{tr} (rad)",
                                                     "Events",
                                                     "ELE_LOGY E1 P");
  h2_ele_dPhiEleClVsEta_propOut = bookH2(iBooker,
                                         "dPhiEleClVsEta_propOut",
                                         "ele #phi_{EleCl} - #phi_{tr} vs eta, prop from out",
                                         eta2D_nbin,
                                         eta_min,
                                         eta_max,
                                         dphimatch2D_nbin,
                                         dphimatch_min,
                                         dphimatch_max);
  h2_ele_dPhiEleClVsPhi_propOut = bookH2(iBooker,
                                         "dPhiEleClVsPhi_propOut",
                                         "ele #phi_{EleCl} - #phi_{tr} vs phi, prop from out",
                                         phi2D_nbin,
                                         phi_min,
                                         phi_max,
                                         dphimatch2D_nbin,
                                         dphimatch_min,
                                         dphimatch_max);
  h2_ele_dPhiEleClVsPt_propOut = bookH2(iBooker,
                                        "dPhiSEleClsPt_propOut",
                                        "ele #phi_{EleCl} - #phi_{tr} vs pt, prop from out",
                                        pt2D_nbin,
                                        0.,
                                        pt_max,
                                        dphimatch2D_nbin,
                                        dphimatch_min,
                                        dphimatch_max);
  h1_ele_HoE = bookH1withSumw2(
      iBooker, "HoE", "ele hadronic energy / em energy", hoe_nbin, hoe_min, hoe_max, "H/E", "Events", "ELE_LOGY E1 P");
  h1_ele_HoE_barrel = bookH1withSumw2(iBooker,
                                      "HoE_barrel",
                                      "ele hadronic energy / em energy, barrel",
                                      hoe_nbin,
                                      hoe_min,
                                      hoe_max,
                                      "H/E",
                                      "Events",
                                      "ELE_LOGY E1 P");
  h1_ele_HoE_endcaps = bookH1withSumw2(iBooker,
                                       "HoE_endcaps",
                                       "ele hadronic energy / em energy, endcaps",
                                       hoe_nbin,
                                       hoe_min,
                                       hoe_max,
                                       "H/E",
                                       "Events",
                                       "ELE_LOGY E1 P");
  h1_ele_HoE_bc = bookH1withSumw2(iBooker,
                                  "HoE_bc",
                                  "ele hadronic energy / em energy behind cluster",
                                  hoe_nbin,
                                  hoe_min,
                                  hoe_max,
                                  "H/E",
                                  "Events",
                                  "ELE_LOGY E1 P");
  h1_ele_HoE_bc_barrel = bookH1withSumw2(iBooker,
                                         "HoE_bc_barrel",
                                         "ele hadronic energy / em energy, behind cluster barrel",
                                         hoe_nbin,
                                         hoe_min,
                                         hoe_max,
                                         "H/E",
                                         "Events",
                                         "ELE_LOGY E1 P");
  h1_ele_HoE_bc_endcaps = bookH1withSumw2(iBooker,
                                          "HoE_bc_endcaps",
                                          "ele hadronic energy / em energy, behind cluster, endcaps",
                                          hoe_nbin,
                                          hoe_min,
                                          hoe_max,
                                          "H/E",
                                          "Events",
                                          "ELE_LOGY E1 P");
  h1_ele_hcalDepth1OverEcalBc = bookH1withSumw2(iBooker,
                                                "hcalDepth1OverEcalBc",
                                                "hcalDepth1OverEcalBc",
                                                hoe_nbin,
                                                hoe_min,
                                                hoe_max,
                                                "H/E",
                                                "Events",
                                                "ELE_LOGY E1 P");
  h1_ele_hcalDepth1OverEcalBc_barrel = bookH1withSumw2(iBooker,
                                                       "hcalDepth1OverEcalBc_barrel",
                                                       "hcalDepth1OverEcalBc_barrel",
                                                       hoe_nbin,
                                                       hoe_min,
                                                       hoe_max,
                                                       "H/E",
                                                       "Events",
                                                       "ELE_LOGY E1 P");
  h1_ele_hcalDepth1OverEcalBc_endcaps = bookH1withSumw2(iBooker,
                                                        "hcalDepth1OverEcalBc_endcaps",
                                                        "hcalDepth1OverEcalBc_endcaps",
                                                        hoe_nbin,
                                                        hoe_min,
                                                        hoe_max,
                                                        "H/E",
                                                        "Events",
                                                        "ELE_LOGY E1 P");
  h1_ele_hcalDepth2OverEcalBc = bookH1withSumw2(iBooker,
                                                "hcalDepth2OverEcalBc",
                                                "hcalDepth2OverEcalBc",
                                                hoe_nbin,
                                                hoe_min,
                                                hoe_max,
                                                "H/E",
                                                "Events",
                                                "ELE_LOGY E1 P");
  h1_ele_hcalDepth2OverEcalBc_barrel = bookH1withSumw2(iBooker,
                                                       "hcalDepth2OverEcalBc_barrel",
                                                       "hcalDepth2OverEcalBc_barrel",
                                                       hoe_nbin,
                                                       hoe_min,
                                                       hoe_max,
                                                       "H/E",
                                                       "Events",
                                                       "ELE_LOGY E1 P");
  h1_ele_hcalDepth2OverEcalBc_endcaps = bookH1withSumw2(iBooker,
                                                        "hcalDepth2OverEcalBc_endcaps",
                                                        "hcalDepth2OverEcalBc_endcaps",
                                                        hoe_nbin,
                                                        hoe_min,
                                                        hoe_max,
                                                        "H/E",
                                                        "Events",
                                                        "ELE_LOGY E1 P");
  h1_ele_HoE_fiducial = bookH1withSumw2(iBooker,
                                        "HoE_fiducial",
                                        "ele hadronic energy / em energy, fiducial region",
                                        hoe_nbin,
                                        hoe_min,
                                        hoe_max,
                                        "H/E",
                                        "Events",
                                        "ELE_LOGY E1 P");
  h2_ele_HoEVsEta = bookH2(iBooker,
                           "HoEVsEta",
                           "ele hadronic energy / em energy vs eta",
                           eta_nbin,
                           eta_min,
                           eta_max,
                           hoe_nbin,
                           hoe_min,
                           hoe_max);
  h2_ele_HoEVsPhi = bookH2(iBooker,
                           "HoEVsPhi",
                           "ele hadronic energy / em energy vs phi",
                           phi2D_nbin,
                           phi_min,
                           phi_max,
                           hoe_nbin,
                           hoe_min,
                           hoe_max);
  h2_ele_HoEVsE =
      bookH2(iBooker, "HoEVsE", "ele hadronic energy / em energy vs E", p_nbin, 0., 300., hoe_nbin, hoe_min, hoe_max);
  setBookPrefix("h_scl");
  //  h1_scl_ESFrac = bookH1withSumw2(iBooker, "ESFrac","Preshower over SC raw energy",100,0.,0.8,"E_{PS} / E^{raw}_{SC}","Events","ELE_LOGY E1 P");
  h1_scl_ESFrac_endcaps = bookH1withSumw2(iBooker,
                                          "ESFrac_endcaps",
                                          "Preshower over SC raw energy , endcaps",
                                          100,
                                          0.,
                                          0.8,
                                          "E_{PS} / E^{raw}_{SC}",
                                          "Events",
                                          "ELE_LOGY E1 P");

  // seeds
  setBookPrefix("h_ele");
  h1_ele_seed_subdet2_ =
      bookH1withSumw2(iBooker, "seedSubdet2", "ele seed subdet 2nd layer", 11, -0.5, 10.5, "2nd hit subdet Id");
  h1_ele_seed_mask_ = bookH1withSumw2(iBooker, "seedMask", "ele seed hits mask", 13, -0.5, 12.5);
  h1_ele_seed_mask_bpix_ =
      bookH1withSumw2(iBooker, "seedMask_Bpix", "ele seed hits mask when subdet2 is bpix", 13, -0.5, 12.5);
  h1_ele_seed_mask_fpix_ =
      bookH1withSumw2(iBooker, "seedMask_Fpix", "ele seed hits mask when subdet2 is bpix", 13, -0.5, 12.5);
  h1_ele_seed_mask_tec_ =
      bookH1withSumw2(iBooker, "seedMask_Tec", "ele seed hits mask when subdet2 is bpix", 13, -0.5, 12.5);
  h1_ele_seed_dphi2_ = bookH1withSumw2(
      iBooker, "seedDphi2", "ele seed dphi 2nd layer", 50, -0.003, +0.003, "#phi_{hit}-#phi_{pred} (rad)");
  h2_ele_seed_dphi2VsEta_ = bookH2(
      iBooker, "seedDphi2_VsEta", "ele seed dphi 2nd layer vs eta", eta2D_nbin, eta_min, eta_max, 50, -0.003, +0.003);
  h2_ele_seed_dphi2VsPt_ =
      bookH2(iBooker, "seedDphi2_VsPt", "ele seed dphi 2nd layer vs pt", pt2D_nbin, 0., pt_max, 50, -0.003, +0.003);
  h1_ele_seed_dphi2pos_ = bookH1withSumw2(
      iBooker, "seedDphi2Pos", "ele seed dphi 2nd layer positron", 50, -0.003, +0.003, "#phi_{hit}-#phi_{pred} (rad)");
  h2_ele_seed_dphi2posVsEta_ = bookH2(iBooker,
                                      "seedDphi2Pos_VsEta",
                                      "ele seed dphi 2nd layer positron vs eta",
                                      eta2D_nbin,
                                      eta_min,
                                      eta_max,
                                      50,
                                      -0.003,
                                      +0.003);
  h2_ele_seed_dphi2posVsPt_ = bookH2(
      iBooker, "seedDphi2Pos_VsPt", "ele seed dphi 2nd layer positron vs pt", pt2D_nbin, 0., pt_max, 50, -0.003, +0.003);
  h1_ele_seed_drz2_ = bookH1withSumw2(
      iBooker, "seedDrz2", "ele seed dr (dz) 2nd layer", 50, -0.03, +0.03, "r(z)_{hit}-r(z)_{pred} (cm)");
  h2_ele_seed_drz2VsEta_ = bookH2(
      iBooker, "seedDrz2_VsEta", "ele seed dr/dz 2nd layer vs eta", eta2D_nbin, eta_min, eta_max, 50, -0.03, +0.03);
  h2_ele_seed_drz2VsPt_ =
      bookH2(iBooker, "seedDrz2_VsPt", "ele seed dr/dz 2nd layer vs pt", pt2D_nbin, 0., pt_max, 50, -0.03, +0.03);
  h1_ele_seed_drz2pos_ = bookH1withSumw2(
      iBooker, "seedDrz2Pos", "ele seed dr (dz) 2nd layer positron", 50, -0.03, +0.03, "r(z)_{hit}-r(z)_{pred} (cm)");
  h2_ele_seed_drz2posVsEta_ = bookH2(iBooker,
                                     "seedDrz2Pos_VsEta",
                                     "ele seed dr/dz 2nd layer positron vs eta",
                                     eta2D_nbin,
                                     eta_min,
                                     eta_max,
                                     50,
                                     -0.03,
                                     +0.03);
  h2_ele_seed_drz2posVsPt_ = bookH2(
      iBooker, "seedDrz2Pos_VsPt", "ele seed dr/dz 2nd layer positron vs pt", pt2D_nbin, 0., pt_max, 50, -0.03, +0.03);

  // classes
  h1_ele_classes = bookH1withSumw2(iBooker, "classes", "ele classes", 20, 0.0, 20., "class Id");
  h1_ele_eta = bookH1withSumw2(iBooker, "eta", "ele electron eta", eta_nbin / 2, 0.0, eta_max);
  h1_ele_eta_golden = bookH1withSumw2(iBooker, "eta_golden", "ele electron eta golden", eta_nbin / 2, 0.0, eta_max);
  h1_ele_eta_bbrem = bookH1withSumw2(iBooker, "eta_bbrem", "ele electron eta bbrem", eta_nbin / 2, 0.0, eta_max);
  h1_ele_eta_shower = bookH1withSumw2(iBooker, "eta_shower", "ele electron eta showering", eta_nbin / 2, 0.0, eta_max);
  h2_ele_PinVsPoutGolden_mode = bookH2(iBooker,
                                       "PinVsPoutGolden_mode",
                                       "ele track inner p vs outer p vs eta, golden, mode of GSF components",
                                       p2D_nbin,
                                       0.,
                                       p_max,
                                       50,
                                       0.,
                                       p_max);
  h2_ele_PinVsPoutShowering_mode = bookH2(iBooker,
                                          "PinVsPoutShowering_mode",
                                          "ele track inner p vs outer p vs eta, showering, mode of GSF components",
                                          p2D_nbin,
                                          0.,
                                          p_max,
                                          50,
                                          0.,
                                          p_max);
  h2_ele_PinVsPoutGolden_mean = bookH2(iBooker,
                                       "PinVsPoutGolden_mean",
                                       "ele track inner p vs outer p vs eta, golden, mean of GSF components",
                                       p2D_nbin,
                                       0.,
                                       p_max,
                                       50,
                                       0.,
                                       p_max);
  h2_ele_PinVsPoutShowering_mean = bookH2(iBooker,
                                          "PinVsPoutShowering_mean",
                                          "ele track inner p vs outer p vs eta, showering, mean of GSF components",
                                          p2D_nbin,
                                          0.,
                                          p_max,
                                          50,
                                          0.,
                                          p_max);
  h2_ele_PtinVsPtoutGolden_mode = bookH2(iBooker,
                                         "PtinVsPtoutGolden_mode",
                                         "ele track inner pt vs outer pt vs eta, golden, mode of GSF components",
                                         pt2D_nbin,
                                         0.,
                                         pt_max,
                                         50,
                                         0.,
                                         pt_max);
  h2_ele_PtinVsPtoutShowering_mode = bookH2(iBooker,
                                            "PtinVsPtoutShowering_mode",
                                            "ele track inner pt vs outer pt vs eta, showering, mode of GSF components",
                                            pt2D_nbin,
                                            0.,
                                            pt_max,
                                            50,
                                            0.,
                                            pt_max);
  h2_ele_PtinVsPtoutGolden_mean = bookH2(iBooker,
                                         "PtinVsPtoutGolden_mean",
                                         "ele track inner pt vs outer pt vs eta, golden, mean of GSF components",
                                         pt2D_nbin,
                                         0.,
                                         pt_max,
                                         50,
                                         0.,
                                         pt_max);
  h2_ele_PtinVsPtoutShowering_mean = bookH2(iBooker,
                                            "PtinVsPtoutShowering_mean",
                                            "ele track inner pt vs outer pt vs eta, showering, mean of GSF components",
                                            pt2D_nbin,
                                            0.,
                                            pt_max,
                                            50,
                                            0.,
                                            pt_max);
  setBookPrefix("h_scl");
  h1_scl_EoEmatchingObjectGolden_barrel = bookH1withSumw2(iBooker,
                                                          "EoEmatchingObject_golden_barrel",
                                                          "ele supercluster energy / gen energy, golden, barrel",
                                                          popmatching_nbin,
                                                          popmatching_min,
                                                          popmatching_max);
  h1_scl_EoEmatchingObjectGolden_endcaps = bookH1withSumw2(iBooker,
                                                           "EoEmatchingObject_golden_endcaps",
                                                           "ele supercluster energy / gen energy, golden, endcaps",
                                                           popmatching_nbin,
                                                           popmatching_min,
                                                           popmatching_max);
  h1_scl_EoEmatchingObjectShowering_barrel = bookH1withSumw2(iBooker,
                                                             "EoEmatchingObject_showering_barrel",
                                                             "ele supercluster energy / gen energy, showering, barrel",
                                                             popmatching_nbin,
                                                             popmatching_min,
                                                             popmatching_max);
  h1_scl_EoEmatchingObjectShowering_endcaps =
      bookH1withSumw2(iBooker,
                      "EoEmatchingObject_showering_endcaps",
                      "ele supercluster energy / gen energy, showering, endcaps",
                      popmatching_nbin,
                      popmatching_min,
                      popmatching_max);
  setBookPrefix("h_ele");

  // isolation
  h1_ele_tkSumPt_dr03 = bookH1withSumw2(iBooker,
                                        "tkSumPt_dr03",
                                        "tk isolation sum, dR=0.3",
                                        100,
                                        0.0,
                                        20.,
                                        "TkIsoSum, cone 0.3 (GeV/c)",
                                        "Events",
                                        "ELE_LOGY E1 P");
  h1_ele_tkSumPt_dr03_barrel = bookH1withSumw2(iBooker,
                                               "tkSumPt_dr03_barrel",
                                               "tk isolation sum, dR=0.3, barrel",
                                               100,
                                               0.0,
                                               20.,
                                               "TkIsoSum, cone 0.3 (GeV/c)",
                                               "Events",
                                               "ELE_LOGY E1 P");
  h1_ele_tkSumPt_dr03_endcaps = bookH1withSumw2(iBooker,
                                                "tkSumPt_dr03_endcaps",
                                                "tk isolation sum, dR=0.3, endcaps",
                                                100,
                                                0.0,
                                                20.,
                                                "TkIsoSum, cone 0.3 (GeV/c)",
                                                "Events",
                                                "ELE_LOGY E1 P");
  h1_ele_ecalRecHitSumEt_dr03 = bookH1withSumw2(iBooker,
                                                "ecalRecHitSumEt_dr03",
                                                "ecal isolation sum, dR=0.3",
                                                100,
                                                0.0,
                                                20.,
                                                "EcalIsoSum, cone 0.3 (GeV)",
                                                "Events",
                                                "ELE_LOGY E1 P");
  h1_ele_ecalRecHitSumEt_dr03_barrel = bookH1withSumw2(iBooker,
                                                       "ecalRecHitSumEt_dr03_barrel",
                                                       "ecal isolation sum, dR=0.3, barrel",
                                                       100,
                                                       0.0,
                                                       20.,
                                                       "EcalIsoSum, cone 0.3 (GeV)",
                                                       "Events",
                                                       "ELE_LOGY E1 P");
  h1_ele_ecalRecHitSumEt_dr03_endcaps = bookH1withSumw2(iBooker,
                                                        "ecalRecHitSumEt_dr03_endcaps",
                                                        "ecal isolation sum, dR=0.3, endcaps",
                                                        100,
                                                        0.0,
                                                        20.,
                                                        "EcalIsoSum, cone 0.3 (GeV)",
                                                        "Events",
                                                        "ELE_LOGY E1 P");
  h1_ele_hcalTowerSumEt_dr03_depth1 = bookH1withSumw2(iBooker,
                                                      "hcalTowerSumEt_dr03_depth1",
                                                      "hcal depth1 isolation sum, dR=0.3",
                                                      100,
                                                      0.0,
                                                      20.,
                                                      "Hcal1IsoSum, cone 0.3 (GeV)",
                                                      "Events",
                                                      "ELE_LOGY E1 P");
  h1_ele_hcalTowerSumEt_dr03_depth1_barrel = bookH1withSumw2(iBooker,
                                                             "hcalTowerSumEt_dr03_depth1_barrel",
                                                             "hcal depth1 isolation sum, dR=0.3, barrel",
                                                             100,
                                                             0.0,
                                                             20.,
                                                             "Hcal1IsoSum, cone 0.3 (GeV)",
                                                             "Events",
                                                             "ELE_LOGY E1 P");
  h1_ele_hcalTowerSumEt_dr03_depth1_endcaps = bookH1withSumw2(iBooker,
                                                              "hcalTowerSumEt_dr03_depth1_endcaps",
                                                              "hcal depth1 isolation sum, dR=0.3, endcaps",
                                                              100,
                                                              0.0,
                                                              20.,
                                                              "Hcal1IsoSum, cone 0.3 (GeV)",
                                                              "Events",
                                                              "ELE_LOGY E1 P");
  h1_ele_hcalTowerSumEt_dr03_depth2 = bookH1withSumw2(iBooker,
                                                      "hcalTowerSumEt_dr03_depth2",
                                                      "hcal depth2 isolation sum, dR=0.3",
                                                      100,
                                                      0.0,
                                                      20.,
                                                      "Hcal2IsoSum, cone 0.3 (GeV)",
                                                      "Events",
                                                      "ELE_LOGY E1 P");
  h1_ele_tkSumPt_dr04 = bookH1withSumw2(iBooker,
                                        "tkSumPt_dr04",
                                        "tk isolation sum, dR=0.4",
                                        100,
                                        0.0,
                                        20.,
                                        "TkIsoSum, cone 0.4 (GeV/c)",
                                        "Events",
                                        "ELE_LOGY E1 P");
  h1_ele_tkSumPt_dr04_barrel = bookH1withSumw2(iBooker,
                                               "tkSumPt_dr04_barrel",
                                               "tk isolation sum, dR=0.4, barrel",
                                               100,
                                               0.0,
                                               20.,
                                               "TkIsoSum, cone 0.4 (GeV/c)",
                                               "Events",
                                               "ELE_LOGY E1 P");
  h1_ele_tkSumPt_dr04_endcaps = bookH1withSumw2(iBooker,
                                                "tkSumPt_dr04_endcaps",
                                                "tk isolation sum, dR=0.4, endcaps",
                                                100,
                                                0.0,
                                                20.,
                                                "TkIsoSum, cone 0.4 (GeV/c)",
                                                "Events",
                                                "ELE_LOGY E1 P");
  h1_ele_ecalRecHitSumEt_dr04 = bookH1withSumw2(iBooker,
                                                "ecalRecHitSumEt_dr04",
                                                "ecal isolation sum, dR=0.4",
                                                100,
                                                0.0,
                                                20.,
                                                "EcalIsoSum, cone 0.4 (GeV)",
                                                "Events",
                                                "ELE_LOGY E1 P");
  h1_ele_ecalRecHitSumEt_dr04_barrel = bookH1withSumw2(iBooker,
                                                       "ecalRecHitSumEt_dr04_barrel",
                                                       "ecal isolation sum, dR=0.4, barrel",
                                                       100,
                                                       0.0,
                                                       20.,
                                                       "EcalIsoSum, cone 0.4 (GeV)",
                                                       "Events",
                                                       "ELE_LOGY E1 P");
  h1_ele_ecalRecHitSumEt_dr04_endcaps = bookH1withSumw2(iBooker,
                                                        "ecalRecHitSumEt_dr04_endcaps",
                                                        "ecal isolation sum, dR=0.4, endcaps",
                                                        100,
                                                        0.0,
                                                        20.,
                                                        "EcalIsoSum, cone 0.4 (GeV)",
                                                        "Events",
                                                        "ELE_LOGY E1 P");
  h1_ele_hcalTowerSumEt_dr04_depth1 = bookH1withSumw2(iBooker,
                                                      "hcalTowerSumEt_dr04_depth1",
                                                      "hcal depth1 isolation sum, dR=0.4",
                                                      100,
                                                      0.0,
                                                      20.,
                                                      "Hcal1IsoSum, cone 0.4 (GeV)",
                                                      "Events",
                                                      "ELE_LOGY E1 P");
  h1_ele_hcalTowerSumEt_dr04_depth1_barrel = bookH1withSumw2(iBooker,
                                                             "hcalTowerSumEt_dr04_depth1_barrel",
                                                             "hcal depth1 isolation sum, dR=0.4, barrel",
                                                             100,
                                                             0.0,
                                                             20.,
                                                             "Hcal1IsoSum, cone 0.4 (GeV)",
                                                             "Events",
                                                             "ELE_LOGY E1 P");
  h1_ele_hcalTowerSumEt_dr04_depth1_endcaps = bookH1withSumw2(iBooker,
                                                              "hcalTowerSumEt_dr04_depth1_endcaps",
                                                              "hcal depth1 isolation sum, dR=0.4, endcaps",
                                                              100,
                                                              0.0,
                                                              20.,
                                                              "Hcal1IsoSum, cone 0.4 (GeV)",
                                                              "Events",
                                                              "ELE_LOGY E1 P");
  h1_ele_hcalTowerSumEt_dr04_depth2 = bookH1withSumw2(iBooker,
                                                      "hcalTowerSumEt_dr04_depth2",
                                                      "hcal depth2 isolation sum, dR=0.4",
                                                      100,
                                                      0.0,
                                                      20.,
                                                      "Hcal2IsoSum, cone 0.4 (GeV)",
                                                      "Events",
                                                      "ELE_LOGY E1 P");

  // newHCAL
  // isolation new hcal
  h1_ele_hcalTowerSumEtBc_dr03_depth1 = bookH1withSumw2(iBooker,
                                                        "hcalTowerSumEtBc_dr03_depth1",
                                                        "hcal depth1 isolation sum behind cluster, dR=0.3",
                                                        100,
                                                        0.0,
                                                        20.,
                                                        "Hcal1IsoSum, cone 0.3 (GeV)",
                                                        "Events",
                                                        "ELE_LOGY E1 P");
  h1_ele_hcalTowerSumEtBc_dr03_depth1_barrel =
      bookH1withSumw2(iBooker,
                      "hcalTowerSumEtBc_dr03_depth1_barrel",
                      "hcal depth1 isolation sum behind cluster, dR=0.3, barrel",
                      100,
                      0.0,
                      20.,
                      "Hcal1IsoSum, cone 0.3 (GeV)",
                      "Events",
                      "ELE_LOGY E1 P");
  h1_ele_hcalTowerSumEtBc_dr03_depth1_endcaps =
      bookH1withSumw2(iBooker,
                      "hcalTowerSumEtBc_dr03_depth1_endcaps",
                      "hcal depth1 isolation sum behind cluster, dR=0.3, endcaps",
                      100,
                      0.0,
                      20.,
                      "Hcal1IsoSum, cone 0.3 (GeV)",
                      "Events",
                      "ELE_LOGY E1 P");

  h1_ele_hcalTowerSumEtBc_dr04_depth1 = bookH1withSumw2(iBooker,
                                                        "hcalTowerSumEtBc_dr04_depth1",
                                                        "hcal depth1 isolation sum behind cluster, dR=0.4",
                                                        100,
                                                        0.0,
                                                        20.,
                                                        "Hcal1IsoSum, cone 0.4 (GeV)",
                                                        "Events",
                                                        "ELE_LOGY E1 P");
  h1_ele_hcalTowerSumEtBc_dr04_depth1_barrel =
      bookH1withSumw2(iBooker,
                      "hcalTowerSumEtBc_dr04_depth1_barrel",
                      "hcal depth1 isolation sum behind cluster, dR=0.4, barrel",
                      100,
                      0.0,
                      20.,
                      "Hcal1IsoSum, cone 0.4 (GeV)",
                      "Events",
                      "ELE_LOGY E1 P");
  h1_ele_hcalTowerSumEtBc_dr04_depth1_endcaps =
      bookH1withSumw2(iBooker,
                      "hcalTowerSumEtBc_dr04_depth1_endcaps",
                      "hcal depth1 isolation sum behind cluster, dR=0.4, endcaps",
                      100,
                      0.0,
                      20.,
                      "Hcal1IsoSum, cone 0.4 (GeV)",
                      "Events",
                      "ELE_LOGY E1 P");

  h1_ele_hcalTowerSumEtBc_dr03_depth2 = bookH1withSumw2(iBooker,
                                                        "hcalTowerSumEtBc_dr03_depth2",
                                                        "hcal depth2 isolation sum behind cluster, dR=0.3",
                                                        100,
                                                        0.0,
                                                        20.,
                                                        "Hcal1IsoSum, cone 0.3 (GeV)",
                                                        "Events",
                                                        "ELE_LOGY E1 P");
  h1_ele_hcalTowerSumEtBc_dr03_depth2_barrel =
      bookH1withSumw2(iBooker,
                      "hcalTowerSumEtBc_dr03_depth2_barrel",
                      "hcal depth2 isolation sum behind cluster, dR=0.3, barrel",
                      100,
                      0.0,
                      20.,
                      "Hcal1IsoSum, cone 0.3 (GeV)",
                      "Events",
                      "ELE_LOGY E1 P");
  h1_ele_hcalTowerSumEtBc_dr03_depth2_endcaps =
      bookH1withSumw2(iBooker,
                      "hcalTowerSumEtBc_dr03_depth2_endcaps",
                      "hcal depth2 isolation sum behind cluster, dR=0.3, endcaps",
                      100,
                      0.0,
                      20.,
                      "Hcal1IsoSum, cone 0.3 (GeV)",
                      "Events",
                      "ELE_LOGY E1 P");

  h1_ele_hcalTowerSumEtBc_dr04_depth2 = bookH1withSumw2(iBooker,
                                                        "hcalTowerSumEtBc_dr04_depth2",
                                                        "hcal depth2 isolation sum behind cluster, dR=0.4",
                                                        100,
                                                        0.0,
                                                        20.,
                                                        "Hcal1IsoSum, cone 0.4 (GeV)",
                                                        "Events",
                                                        "ELE_LOGY E1 P");
  h1_ele_hcalTowerSumEtBc_dr04_depth2_barrel =
      bookH1withSumw2(iBooker,
                      "hcalTowerSumEtBc_dr04_depth2_barrel",
                      "hcal depth2 isolation sum behind cluster, dR=0.4, barrel",
                      100,
                      0.0,
                      20.,
                      "Hcal1IsoSum, cone 0.4 (GeV)",
                      "Events",
                      "ELE_LOGY E1 P");
  h1_ele_hcalTowerSumEtBc_dr04_depth2_endcaps =
      bookH1withSumw2(iBooker,
                      "hcalTowerSumEtBc_dr04_depth2_endcaps",
                      "hcal depth2 isolation sum behind cluster, dR=0.4, endcaps",
                      100,
                      0.0,
                      20.,
                      "Hcal1IsoSum, cone 0.4 (GeV)",
                      "Events",
                      "ELE_LOGY E1 P");

  // fbrem
  h1_ele_fbrem = bookH1withSumw2(
      iBooker, "fbrem", "ele brem fraction, mode of GSF components", 100, 0., 1., "P_{in} - P_{out} / P_{in}");
  h1_ele_fbrem_barrel = bookH1withSumw2(iBooker,
                                        "fbrem_barrel",
                                        "ele brem fraction for barrel, mode of GSF components",
                                        100,
                                        0.,
                                        1.,
                                        "P_{in} - P_{out} / P_{in}");
  h1_ele_fbrem_endcaps = bookH1withSumw2(iBooker,
                                         "fbrem_endcaps",
                                         "ele brem franction for endcaps, mode of GSF components",
                                         100,
                                         0.,
                                         1.,
                                         "P_{in} - P_{out} / P_{in}");
  p1_ele_fbremVsEta_mode = bookP1(iBooker,
                                  "fbremvsEtamode",
                                  "mean ele brem fraction vs eta, mode of GSF components",
                                  eta2D_nbin,
                                  eta_min,
                                  eta_max,
                                  0.,
                                  1.,
                                  "#eta",
                                  "<P_{in} - P_{out} / P_{in}>");
  p1_ele_fbremVsEta_mean = bookP1(iBooker,
                                  "fbremvsEtamean",
                                  "mean ele brem fraction vs eta, mean of GSF components",
                                  eta2D_nbin,
                                  eta_min,
                                  eta_max,
                                  0.,
                                  1.,
                                  "#eta",
                                  "<P_{in} - P_{out} / P_{in}>");
  h1_ele_superclusterfbrem =
      bookH1withSumw2(iBooker, "superclusterfbrem", "supercluster brem fraction", 100, 0., 1., "1 - E_{ele} / E_{SC}");
  h1_ele_superclusterfbrem_barrel = bookH1withSumw2(
      iBooker, "superclusterfbrem_barrel", "supercluster brem fraction for barrel", 100, 0., 1., "1 - E_{ele} / E_{SC}");
  h1_ele_superclusterfbrem_endcaps = bookH1withSumw2(iBooker,
                                                     "superclusterfbrem_endcaps",
                                                     "supercluster brem franction for endcaps",
                                                     100,
                                                     0.,
                                                     1.,
                                                     "1 - E_{ele} / E_{SC}");
  // e/g et pflow electrons
  h1_ele_mva = bookH1withSumw2(iBooker, "mva", "ele identification mva", 100, -1., 1.);
  h1_ele_mva_barrel = bookH1withSumw2(iBooker, "mva_barrel", "ele identification mva barrel", 100, -1., 1.);
  h1_ele_mva_endcaps = bookH1withSumw2(iBooker, "mva_endcaps", "ele identification mva endcaps", 100, -1., 1.);
  h1_ele_mva_isolated = bookH1withSumw2(iBooker, "mva_isolated", "ele identification mva isolated", 100, -1., 1.);
  h1_ele_mva_barrel_isolated =
      bookH1withSumw2(iBooker, "mva_isolated_barrel", "ele identification mva isolated barrel", 100, -1., 1.);
  h1_ele_mva_endcaps_isolated =
      bookH1withSumw2(iBooker, "mva_isolated_endcaps", "ele identification mva isolated endcaps", 100, -1., 1.);
  h1_ele_provenance = bookH1withSumw2(iBooker, "provenance", "ele provenance", 5, -2., 3.);
  h1_ele_provenance_barrel = bookH1withSumw2(iBooker, "provenance_barrel", "ele provenance barrel", 5, -2., 3.);
  h1_ele_provenance_endcaps = bookH1withSumw2(iBooker, "provenance_endcaps", "ele provenance endcaps", 5, -2., 3.);

  // pflow isolation variables
  h1_ele_chargedHadronIso = bookH1withSumw2(
      iBooker, "chargedHadronIso", "chargedHadronIso", 100, 0.0, 20., "chargedHadronIso", "Events", "ELE_LOGY E1 P");
  h1_ele_chargedHadronIso_barrel = bookH1withSumw2(iBooker,
                                                   "chargedHadronIso_barrel",
                                                   "chargedHadronIso for barrel",
                                                   100,
                                                   0.0,
                                                   20.,
                                                   "chargedHadronIso_barrel",
                                                   "Events",
                                                   "ELE_LOGY E1 P");
  h1_ele_chargedHadronIso_endcaps = bookH1withSumw2(iBooker,
                                                    "chargedHadronIso_endcaps",
                                                    "chargedHadronIso for endcaps",
                                                    100,
                                                    0.0,
                                                    20.,
                                                    "chargedHadronIso_endcaps",
                                                    "Events",
                                                    "ELE_LOGY E1 P");
  h1_ele_neutralHadronIso = bookH1withSumw2(
      iBooker, "neutralHadronIso", "neutralHadronIso", 21, 0.0, 20., "neutralHadronIso", "Events", "ELE_LOGY E1 P");
  h1_ele_neutralHadronIso_barrel = bookH1withSumw2(iBooker,
                                                   "neutralHadronIso_barrel",
                                                   "neutralHadronIso for barrel",
                                                   21,
                                                   0.0,
                                                   20.,
                                                   "neutralHadronIso_barrel",
                                                   "Events",
                                                   "ELE_LOGY E1 P");
  h1_ele_neutralHadronIso_endcaps = bookH1withSumw2(iBooker,
                                                    "neutralHadronIso_endcaps",
                                                    "neutralHadronIso for endcaps",
                                                    21,
                                                    0.0,
                                                    20.,
                                                    "neutralHadronIso_endcaps",
                                                    "Events",
                                                    "ELE_LOGY E1 P");
  h1_ele_photonIso =
      bookH1withSumw2(iBooker, "photonIso", "photonIso", 100, 0.0, 20., "photonIso", "Events", "ELE_LOGY E1 P");
  h1_ele_photonIso_barrel = bookH1withSumw2(
      iBooker, "photonIso_barrel", "photonIso for barrel", 100, 0.0, 20., "photonIso_barrel", "Events", "ELE_LOGY E1 P");
  h1_ele_photonIso_endcaps = bookH1withSumw2(iBooker,
                                             "photonIso_endcaps",
                                             "photonIso for endcaps",
                                             100,
                                             0.0,
                                             20.,
                                             "photonIso_endcaps",
                                             "Events",
                                             "ELE_LOGY E1 P");
  // -- pflow over pT
  h1_ele_chargedHadronRelativeIso = bookH1withSumw2(iBooker,
                                                    "chargedHadronRelativeIso",
                                                    "chargedHadronRelativeIso",
                                                    100,
                                                    0.0,
                                                    2.,
                                                    "chargedHadronRelativeIso",
                                                    "Events",
                                                    "ELE_LOGY E1 P");
  h1_ele_chargedHadronRelativeIso_barrel = bookH1withSumw2(iBooker,
                                                           "chargedHadronRelativeIso_barrel",
                                                           "chargedHadronRelativeIso for barrel",
                                                           100,
                                                           0.0,
                                                           2.,
                                                           "chargedHadronRelativeIso_barrel",
                                                           "Events",
                                                           "ELE_LOGY E1 P");
  h1_ele_chargedHadronRelativeIso_endcaps = bookH1withSumw2(iBooker,
                                                            "chargedHadronRelativeIso_endcaps",
                                                            "chargedHadronRelativeIso for endcaps",
                                                            100,
                                                            0.0,
                                                            2.,
                                                            "chargedHadronRelativeIso_endcaps",
                                                            "Events",
                                                            "ELE_LOGY E1 P");
  h1_ele_neutralHadronRelativeIso = bookH1withSumw2(iBooker,
                                                    "neutralHadronRelativeIso",
                                                    "neutralHadronRelativeIso",
                                                    100,
                                                    0.0,
                                                    2.,
                                                    "neutralHadronRelativeIso",
                                                    "Events",
                                                    "ELE_LOGY E1 P");
  h1_ele_neutralHadronRelativeIso_barrel = bookH1withSumw2(iBooker,
                                                           "neutralHadronRelativeIso_barrel",
                                                           "neutralHadronRelativeIso for barrel",
                                                           100,
                                                           0.0,
                                                           2.,
                                                           "neutralHadronRelativeIso_barrel",
                                                           "Events",
                                                           "ELE_LOGY E1 P");
  h1_ele_neutralHadronRelativeIso_endcaps = bookH1withSumw2(iBooker,
                                                            "neutralHadronRelativeIso_endcaps",
                                                            "neutralHadronRelativeIso for endcaps",
                                                            100,
                                                            0.0,
                                                            2.,
                                                            "neutralHadronRelativeIso_endcaps",
                                                            "Events",
                                                            "ELE_LOGY E1 P");
  h1_ele_photonRelativeIso = bookH1withSumw2(
      iBooker, "photonRelativeIso", "photonRelativeIso", 100, 0.0, 2., "photonRelativeIso", "Events", "ELE_LOGY E1 P");
  h1_ele_photonRelativeIso_barrel = bookH1withSumw2(iBooker,
                                                    "photonRelativeIso_barrel",
                                                    "photonRelativeIso for barrel",
                                                    100,
                                                    0.0,
                                                    2.,
                                                    "photonRelativeIso_barrel",
                                                    "Events",
                                                    "ELE_LOGY E1 P");
  h1_ele_photonRelativeIso_endcaps = bookH1withSumw2(iBooker,
                                                     "photonRelativeIso_endcaps",
                                                     "photonRelativeIso for endcaps",
                                                     100,
                                                     0.0,
                                                     2.,
                                                     "photonRelativeIso_endcaps",
                                                     "Events",
                                                     "ELE_LOGY E1 P");

  // conversion rejection information
  h1_ele_convFlags = bookH1withSumw2(iBooker, "convFlags", "conversion rejection flag", 5, -2.5, 2.5);
  h1_ele_convFlags_all =
      bookH1withSumw2(iBooker, "convFlags_all", "conversion rejection flag, all electrons", 5, -2.5, 2.5);
  h1_ele_convDist = bookH1withSumw2(iBooker, "convDist", "distance to the conversion partner", 100, -15., 15.);
  h1_ele_convDist_all =
      bookH1withSumw2(iBooker, "convDist_all", "distance to the conversion partner, all electrons", 100, -15., 15.);
  h1_ele_convDcot = bookH1withSumw2(
      iBooker, "convDcot", "difference of cot(angle) with the conversion partner", 100, -CLHEP::pi / 2., CLHEP::pi / 2.);
  h1_ele_convDcot_all = bookH1withSumw2(iBooker,
                                        "convDcot_all",
                                        "difference of cot(angle) with the conversion partner, all electrons",
                                        100,
                                        -CLHEP::pi / 2.,
                                        CLHEP::pi / 2.);
  h1_ele_convRadius = bookH1withSumw2(iBooker, "convRadius", "signed conversion radius", 100, 0., 130.);
  h1_ele_convRadius_all =
      bookH1withSumw2(iBooker, "convRadius_all", "signed conversion radius, all electrons", 100, 0., 130.);
}

ElectronMcFakeValidator::~ElectronMcFakeValidator() {}

//=========================================================================
// Main methods
//=========================================================================

void ElectronMcFakeValidator::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  // get reco electrons
  auto gsfElectrons = iEvent.getHandle(electronCollection_);
  auto gsfElectronCores = iEvent.getHandle(electronCoreCollection_);
  auto gsfElectronTracks = iEvent.getHandle(electronTrackCollection_);
  auto gsfElectronSeeds = iEvent.getHandle(electronSeedCollection_);

  auto isoFromDepsTk03Handle = iEvent.getHandle(isoFromDepsTk03Tag_);
  auto isoFromDepsTk04Handle = iEvent.getHandle(isoFromDepsTk04Tag_);
  auto isoFromDepsEcalFull03Handle = iEvent.getHandle(isoFromDepsEcalFull03Tag_);
  auto isoFromDepsEcalFull04Handle = iEvent.getHandle(isoFromDepsEcalFull04Tag_);
  auto isoFromDepsEcalReduced03Handle = iEvent.getHandle(isoFromDepsEcalReduced03Tag_);
  auto isoFromDepsEcalReduced04Handle = iEvent.getHandle(isoFromDepsEcalReduced04Tag_);
  auto isoFromDepsHcal03Handle = iEvent.getHandle(isoFromDepsHcal03Tag_);
  auto isoFromDepsHcal04Handle = iEvent.getHandle(isoFromDepsHcal04Tag_);

  /*edm::Handle<reco::VertexCollection> vertexCollectionHandle;
  iEvent.getByToken(offlineVerticesCollection_, vertexCollectionHandle);*/
  auto vertexCollectionHandle = iEvent.getHandle(offlineVerticesCollection_);
  if (!vertexCollectionHandle.isValid()) {
    edm::LogInfo("ElectronMcFakeValidator::analyze") << "vertexCollectionHandle KO";
  } else {
    edm::LogInfo("ElectronMcFakeValidator::analyze") << "vertexCollectionHandle OK";
  }

  // get gen jets
  auto genJets = iEvent.getHandle(matchingObjectCollection_);

  // get the beamspot from the Event:
  auto recoBeamSpotHandle = iEvent.getHandle(beamSpotTag_);
  const BeamSpot bs = *recoBeamSpotHandle;

  edm::LogInfo("ElectronMcFakeValidator::analyze")
      << "Treating event " << iEvent.id() << " with " << gsfElectrons.product()->size() << " electrons";
  h1_recEleNum_->Fill((*gsfElectrons).size());
  h1_recCoreNum_->Fill((*gsfElectronCores).size());
  h1_recTrackNum_->Fill((*gsfElectronTracks).size());
  h1_recSeedNum_->Fill((*gsfElectronSeeds).size());
  h1_recOfflineVertices_->Fill((*vertexCollectionHandle).size());

  // all rec electrons
  reco::GsfElectronCollection::const_iterator gsfIter;
  for (gsfIter = gsfElectrons->begin(); gsfIter != gsfElectrons->end(); gsfIter++) {
    // preselect electrons
    if (gsfIter->pt() > maxPt_ || std::abs(gsfIter->eta()) > maxAbsEta_) {
      continue;
    }

    h1_ele_EoverP_all->Fill(gsfIter->eSuperClusterOverP());
    h1_ele_EseedOP_all->Fill(gsfIter->eSeedClusterOverP());
    h1_ele_EoPout_all->Fill(gsfIter->eSeedClusterOverPout());
    h1_ele_EeleOPout_all->Fill(gsfIter->eEleClusterOverPout());
    h1_ele_dEtaSc_propVtx_all->Fill(gsfIter->deltaEtaSuperClusterTrackAtVtx());
    h1_ele_dPhiSc_propVtx_all->Fill(gsfIter->deltaPhiSuperClusterTrackAtVtx());
    h1_ele_dEtaCl_propOut_all->Fill(gsfIter->deltaEtaSeedClusterTrackAtCalo());
    h1_ele_dPhiCl_propOut_all->Fill(gsfIter->deltaPhiSeedClusterTrackAtCalo());
    h1_ele_HoE_all->Fill(gsfIter->hadronicOverEm());
    h1_ele_HoE_bc_all->Fill(gsfIter->hcalOverEcalBc());
    double d = gsfIter->vertex().x() * gsfIter->vertex().x() + gsfIter->vertex().y() * gsfIter->vertex().y();
    h1_ele_TIP_all->Fill(sqrt(d));
    h1_ele_vertexEta_all->Fill(gsfIter->eta());
    h1_ele_vertexPt_all->Fill(gsfIter->pt());
    float enrj1 = gsfIter->ecalEnergy();

    // mee
    reco::GsfElectronCollection::const_iterator gsfIter2;
    for (gsfIter2 = gsfIter + 1; gsfIter2 != gsfElectrons->end(); gsfIter2++) {
      math::XYZTLorentzVector p12 = (*gsfIter).p4() + (*gsfIter2).p4();
      float mee2 = p12.Dot(p12);
      h1_ele_mee_all->Fill(sqrt(mee2));
      float enrj2 = gsfIter2->ecalEnergy();
      h2_ele_E2mnE1vsMee_all->Fill(sqrt(mee2), enrj2 - enrj1);
      if (gsfIter->ecalDrivenSeed() && gsfIter2->ecalDrivenSeed()) {
        h2_ele_E2mnE1vsMee_egeg_all->Fill(sqrt(mee2), enrj2 - enrj1);
      }
      if (gsfIter->charge() * gsfIter2->charge() < 0.) {
        h1_ele_mee_os->Fill(sqrt(mee2));
      }
    }

    // conversion rejection
    int flags = gsfIter->convFlags();
    if (flags == -9999) {
      flags = -1;
    }
    h1_ele_convFlags_all->Fill(flags);
    if (flags >= 0.) {
      h1_ele_convDist_all->Fill(gsfIter->convDist());
      h1_ele_convDcot_all->Fill(gsfIter->convDcot());
      h1_ele_convRadius_all->Fill(gsfIter->convRadius());
    }
  }

  // association matching object-reco electrons
  int matchingObjectNum = 0;
  reco::GenJetCollection::const_iterator moIter;
  for (moIter = genJets->begin(); moIter != genJets->end(); ++moIter) {
    // number of matching objects
    matchingObjectNum++;

    if (moIter->energy() / cosh(moIter->eta()) > maxPt_ || std::abs(moIter->eta()) > maxAbsEta_) {
      continue;
    }

    // suppress the endcaps
    //if (std::abs(moIter->eta()) > 1.5) continue;
    // select central z
    //if ( std::abs((*mcIter)->production_vertex()->position().z())>50.) continue;

    h1_matchingObjectEta->Fill(moIter->eta());
    h1_matchingObjectAbsEta->Fill(std::abs(moIter->eta()));
    h1_matchingObjectP->Fill(moIter->energy());
    h1_matchingObjectPt->Fill(moIter->energy() / cosh(moIter->eta()));
    h1_matchingObjectPhi->Fill(moIter->phi());
    h1_matchingObjectZ->Fill(moIter->vz());

    // looking for the best matching gsf electron
    bool okGsfFound = false;
    double gsfOkRatio = 999999.;

    // find best matched electron
    reco::GsfElectron bestGsfElectron;
    reco::GsfElectronRef bestGsfElectronRef;
    reco::GsfElectronCollection::const_iterator gsfIter;
    reco::GsfElectronCollection::size_type iElectron;
    for (gsfIter = gsfElectrons->begin(), iElectron = 0; gsfIter != gsfElectrons->end(); gsfIter++, iElectron++) {
      double dphi = gsfIter->phi() - moIter->phi();
      if (std::abs(dphi) > CLHEP::pi) {
        dphi = dphi < 0 ? (CLHEP::twopi) + dphi : dphi - CLHEP::twopi;
      }
      double deltaR = sqrt(pow((gsfIter->eta() - moIter->eta()), 2) + pow(dphi, 2));
      if (deltaR < deltaR_) {
        //if ( (genPc->pdg_id() == 11) && (gsfIter->charge() < 0.) || (genPc->pdg_id() == -11) &&
        //(gsfIter->charge() > 0.) ){
        double tmpGsfRatio = gsfIter->p() / moIter->energy();
        if (std::abs(tmpGsfRatio - 1) < std::abs(gsfOkRatio - 1)) {
          gsfOkRatio = tmpGsfRatio;
          bestGsfElectronRef = reco::GsfElectronRef(gsfElectrons, iElectron);
          bestGsfElectron = *gsfIter;
          okGsfFound = true;
        }
        //}
      }
    }  // loop over rec ele to look for the best one

    // analysis when the matching object is matched by a rec electron
    if (okGsfFound) {
      // electron related distributions
      h1_ele_charge->Fill(bestGsfElectron.charge());
      h2_ele_chargeVsEta->Fill(bestGsfElectron.eta(), bestGsfElectron.charge());
      h2_ele_chargeVsPhi->Fill(bestGsfElectron.phi(), bestGsfElectron.charge());
      h2_ele_chargeVsPt->Fill(bestGsfElectron.pt(), bestGsfElectron.charge());
      h1_ele_vertexP->Fill(bestGsfElectron.p());
      h1_ele_vertexPt->Fill(bestGsfElectron.pt());
      h2_ele_vertexPtVsEta->Fill(bestGsfElectron.eta(), bestGsfElectron.pt());
      h2_ele_vertexPtVsPhi->Fill(bestGsfElectron.phi(), bestGsfElectron.pt());
      h1_ele_vertexEta->Fill(bestGsfElectron.eta());
      // generated distributions for matched electrons
      h1_ele_matchingObjectPt_matched->Fill(moIter->energy() / cosh(moIter->eta()));
      h1_ele_matchingObjectPhi_matched->Fill(moIter->phi());
      h1_ele_matchingObjectAbsEta_matched->Fill(std::abs(moIter->eta()));
      h1_ele_matchingObjectEta_matched->Fill(moIter->eta());
      h2_ele_vertexEtaVsPhi->Fill(bestGsfElectron.phi(), bestGsfElectron.eta());
      h1_ele_vertexPhi->Fill(bestGsfElectron.phi());
      h1_ele_vertexX->Fill(bestGsfElectron.vertex().x());
      h1_ele_vertexY->Fill(bestGsfElectron.vertex().y());
      h1_ele_vertexZ->Fill(bestGsfElectron.vertex().z());
      h1_ele_matchingObjectZ_matched->Fill(moIter->vz());
      double d =
          (bestGsfElectron.vertex().x() - bs.position().x()) * (bestGsfElectron.vertex().x() - bs.position().x()) +
          (bestGsfElectron.vertex().y() - bs.position().y()) * (bestGsfElectron.vertex().y() - bs.position().y());
      d = sqrt(d);
      h1_ele_vertexTIP->Fill(d);
      h2_ele_vertexTIPVsEta->Fill(bestGsfElectron.eta(), d);
      h2_ele_vertexTIPVsPhi->Fill(bestGsfElectron.phi(), d);
      h2_ele_vertexTIPVsPt->Fill(bestGsfElectron.pt(), d);
      h1_ele_EtaMnEtamatchingObject->Fill(bestGsfElectron.eta() - moIter->eta());
      h2_ele_EtaMnEtamatchingObjectVsEta->Fill(bestGsfElectron.eta(), bestGsfElectron.eta() - moIter->eta());
      h2_ele_EtaMnEtamatchingObjectVsPhi->Fill(bestGsfElectron.phi(), bestGsfElectron.eta() - moIter->eta());
      h2_ele_EtaMnEtamatchingObjectVsPt->Fill(bestGsfElectron.pt(), bestGsfElectron.eta() - moIter->eta());
      h1_ele_PhiMnPhimatchingObject->Fill(bestGsfElectron.phi() - moIter->phi());
      h1_ele_PhiMnPhimatchingObject2->Fill(bestGsfElectron.phi() - moIter->phi());
      h2_ele_PhiMnPhimatchingObjectVsEta->Fill(bestGsfElectron.eta(), bestGsfElectron.phi() - moIter->phi());
      h2_ele_PhiMnPhimatchingObjectVsPhi->Fill(bestGsfElectron.phi(), bestGsfElectron.phi() - moIter->phi());
      h2_ele_PhiMnPhimatchingObjectVsPt->Fill(bestGsfElectron.pt(), bestGsfElectron.phi() - moIter->phi());
      h1_ele_PoPmatchingObject->Fill(bestGsfElectron.p() / moIter->energy());
      h2_ele_PoPmatchingObjectVsEta->Fill(bestGsfElectron.eta(), bestGsfElectron.p() / moIter->energy());
      h2_ele_PoPmatchingObjectVsPhi->Fill(bestGsfElectron.phi(), bestGsfElectron.p() / moIter->energy());
      h2_ele_PoPmatchingObjectVsPt->Fill(bestGsfElectron.py(), bestGsfElectron.p() / moIter->energy());
      if (bestGsfElectron.isEB())
        h1_ele_PoPmatchingObject_barrel->Fill(bestGsfElectron.p() / moIter->energy());
      if (bestGsfElectron.isEE())
        h1_ele_PoPmatchingObject_endcaps->Fill(bestGsfElectron.p() / moIter->energy());

      // supercluster related distributions
      reco::SuperClusterRef sclRef = bestGsfElectron.superCluster();
      if (!bestGsfElectron.ecalDrivenSeed() && bestGsfElectron.trackerDrivenSeed())
        sclRef = bestGsfElectron.parentSuperCluster();
      if (sclRef.isNonnull()) {
        h1_scl_En_->Fill(sclRef->energy());
        double R = TMath::Sqrt(sclRef->x() * sclRef->x() + sclRef->y() * sclRef->y() + sclRef->z() * sclRef->z());
        double Rt = TMath::Sqrt(sclRef->x() * sclRef->x() + sclRef->y() * sclRef->y());
        h1_scl_Et_->Fill(sclRef->energy() * (Rt / R));
        h2_scl_EtVsEta_->Fill(sclRef->eta(), sclRef->energy() * (Rt / R));
        h2_scl_EtVsPhi_->Fill(sclRef->phi(), sclRef->energy() * (Rt / R));
        if (bestGsfElectron.isEB())
          h1_scl_EoEmatchingObject_barrel->Fill(sclRef->energy() / moIter->energy());
        if (bestGsfElectron.isEE())
          h1_scl_EoEmatchingObject_endcaps->Fill(sclRef->energy() / moIter->energy());
        h1_scl_Eta_->Fill(sclRef->eta());
        h2_scl_EtaVsPhi_->Fill(sclRef->phi(), sclRef->eta());
        h1_scl_Phi_->Fill(sclRef->phi());
        /*New from 06 05 2016*/
        //        h1_scl_ESFrac->Fill( sclRef->preshowerEnergy() / sclRef->rawEnergy() );
        if (bestGsfElectron.isEE())
          h1_scl_ESFrac_endcaps->Fill(sclRef->preshowerEnergy() / sclRef->rawEnergy());
      }
      h1_scl_SigIEtaIEta_->Fill(bestGsfElectron.scSigmaIEtaIEta());
      if (bestGsfElectron.isEB())
        h1_scl_SigIEtaIEta_barrel_->Fill(bestGsfElectron.scSigmaIEtaIEta());
      if (bestGsfElectron.isEE())
        h1_scl_SigIEtaIEta_endcaps_->Fill(bestGsfElectron.scSigmaIEtaIEta());
      h1_scl_full5x5_sigmaIetaIeta_->Fill(bestGsfElectron.full5x5_sigmaIetaIeta());
      if (bestGsfElectron.isEB())
        h1_scl_full5x5_sigmaIetaIeta_barrel_->Fill(bestGsfElectron.full5x5_sigmaIetaIeta());
      if (bestGsfElectron.isEE())
        h1_scl_full5x5_sigmaIetaIeta_endcaps_->Fill(bestGsfElectron.full5x5_sigmaIetaIeta());
      h1_scl_E1x5_->Fill(bestGsfElectron.scE1x5());
      if (bestGsfElectron.isEB())
        h1_scl_E1x5_barrel_->Fill(bestGsfElectron.scE1x5());
      if (bestGsfElectron.isEE())
        h1_scl_E1x5_endcaps_->Fill(bestGsfElectron.scE1x5());
      h1_scl_E2x5max_->Fill(bestGsfElectron.scE2x5Max());
      if (bestGsfElectron.isEB())
        h1_scl_E2x5max_barrel_->Fill(bestGsfElectron.scE2x5Max());
      if (bestGsfElectron.isEE())
        h1_scl_E2x5max_endcaps_->Fill(bestGsfElectron.scE2x5Max());
      h1_scl_E5x5_->Fill(bestGsfElectron.scE5x5());
      if (bestGsfElectron.isEB())
        h1_scl_E5x5_barrel_->Fill(bestGsfElectron.scE5x5());
      if (bestGsfElectron.isEE())
        h1_scl_E5x5_endcaps_->Fill(bestGsfElectron.scE5x5());

      // track related distributions
      h1_ele_ambiguousTracks->Fill(bestGsfElectron.ambiguousGsfTracksSize());
      h2_ele_ambiguousTracksVsEta->Fill(bestGsfElectron.eta(), bestGsfElectron.ambiguousGsfTracksSize());
      h2_ele_ambiguousTracksVsPhi->Fill(bestGsfElectron.phi(), bestGsfElectron.ambiguousGsfTracksSize());
      h2_ele_ambiguousTracksVsPt->Fill(bestGsfElectron.pt(), bestGsfElectron.ambiguousGsfTracksSize());
      if (!readAOD_) {  // track extra does not exist in AOD
        h1_ele_foundHits->Fill(bestGsfElectron.gsfTrack()->numberOfValidHits());
        h2_ele_foundHitsVsEta->Fill(bestGsfElectron.eta(), bestGsfElectron.gsfTrack()->numberOfValidHits());
        h2_ele_foundHitsVsPhi->Fill(bestGsfElectron.phi(), bestGsfElectron.gsfTrack()->numberOfValidHits());
        h2_ele_foundHitsVsPt->Fill(bestGsfElectron.pt(), bestGsfElectron.gsfTrack()->numberOfValidHits());
        h1_ele_lostHits->Fill(bestGsfElectron.gsfTrack()->numberOfLostHits());
        h2_ele_lostHitsVsEta->Fill(bestGsfElectron.eta(), bestGsfElectron.gsfTrack()->numberOfLostHits());
        h2_ele_lostHitsVsPhi->Fill(bestGsfElectron.phi(), bestGsfElectron.gsfTrack()->numberOfLostHits());
        h2_ele_lostHitsVsPt->Fill(bestGsfElectron.pt(), bestGsfElectron.gsfTrack()->numberOfLostHits());
        h1_ele_chi2->Fill(bestGsfElectron.gsfTrack()->normalizedChi2());
        h2_ele_chi2VsEta->Fill(bestGsfElectron.eta(), bestGsfElectron.gsfTrack()->normalizedChi2());
        h2_ele_chi2VsPhi->Fill(bestGsfElectron.phi(), bestGsfElectron.gsfTrack()->normalizedChi2());
        h2_ele_chi2VsPt->Fill(bestGsfElectron.pt(), bestGsfElectron.gsfTrack()->normalizedChi2());
      }
      // from gsf track interface, hence using mean
      if (!readAOD_) {  // track extra does not exist in AOD
        h1_ele_PinMnPout->Fill(bestGsfElectron.gsfTrack()->innerMomentum().R() -
                               bestGsfElectron.gsfTrack()->outerMomentum().R());
        h1_ele_outerP->Fill(bestGsfElectron.gsfTrack()->outerMomentum().R());
        h1_ele_outerPt->Fill(bestGsfElectron.gsfTrack()->outerMomentum().Rho());
      }
      // from electron interface, hence using mode
      h1_ele_PinMnPout_mode->Fill(bestGsfElectron.trackMomentumAtVtx().R() - bestGsfElectron.trackMomentumOut().R());
      h2_ele_PinMnPoutVsEta_mode->Fill(
          bestGsfElectron.eta(), bestGsfElectron.trackMomentumAtVtx().R() - bestGsfElectron.trackMomentumOut().R());
      h2_ele_PinMnPoutVsPhi_mode->Fill(
          bestGsfElectron.phi(), bestGsfElectron.trackMomentumAtVtx().R() - bestGsfElectron.trackMomentumOut().R());
      h2_ele_PinMnPoutVsPt_mode->Fill(
          bestGsfElectron.pt(), bestGsfElectron.trackMomentumAtVtx().R() - bestGsfElectron.trackMomentumOut().R());
      h2_ele_PinMnPoutVsE_mode->Fill(bestGsfElectron.caloEnergy(),
                                     bestGsfElectron.trackMomentumAtVtx().R() - bestGsfElectron.trackMomentumOut().R());
      if (!readAOD_)  // track extra does not exist in AOD
        h2_ele_PinMnPoutVsChi2_mode->Fill(
            bestGsfElectron.gsfTrack()->normalizedChi2(),
            bestGsfElectron.trackMomentumAtVtx().R() - bestGsfElectron.trackMomentumOut().R());
      h1_ele_outerP_mode->Fill(bestGsfElectron.trackMomentumOut().R());
      h2_ele_outerPVsEta_mode->Fill(bestGsfElectron.eta(), bestGsfElectron.trackMomentumOut().R());
      h1_ele_outerPt_mode->Fill(bestGsfElectron.trackMomentumOut().Rho());
      h2_ele_outerPtVsEta_mode->Fill(bestGsfElectron.eta(), bestGsfElectron.trackMomentumOut().Rho());
      h2_ele_outerPtVsPhi_mode->Fill(bestGsfElectron.phi(), bestGsfElectron.trackMomentumOut().Rho());
      h2_ele_outerPtVsPt_mode->Fill(bestGsfElectron.pt(), bestGsfElectron.trackMomentumOut().Rho());

      if (!readAOD_) {  // track extra does not exist in AOD
        edm::RefToBase<TrajectorySeed> seed = bestGsfElectron.gsfTrack()->extra()->seedRef();
        ElectronSeedRef elseed = seed.castTo<ElectronSeedRef>();
        h1_ele_seed_subdet2_->Fill(elseed->subDet(1));
        h1_ele_seed_mask_->Fill(elseed->hitsMask());
        if (elseed->subDet(1) == 1) {
          h1_ele_seed_mask_bpix_->Fill(elseed->hitsMask());
        } else if (elseed->subDet(1) == 2) {
          h1_ele_seed_mask_fpix_->Fill(elseed->hitsMask());
        } else if (elseed->subDet(1) == 6) {
          h1_ele_seed_mask_tec_->Fill(elseed->hitsMask());
        }

        if (elseed->dPhiNeg(1) != std::numeric_limits<float>::infinity()) {
          h1_ele_seed_dphi2_->Fill(elseed->dPhiNeg(1));
          h2_ele_seed_dphi2VsEta_->Fill(bestGsfElectron.eta(), elseed->dPhiNeg(1));
          h2_ele_seed_dphi2VsPt_->Fill(bestGsfElectron.pt(), elseed->dPhiNeg(1));
        }
        if (elseed->dPhiPos(1) != std::numeric_limits<float>::infinity()) {
          h1_ele_seed_dphi2pos_->Fill(elseed->dPhiPos(1));
          h2_ele_seed_dphi2posVsEta_->Fill(bestGsfElectron.eta(), elseed->dPhiPos(1));
          h2_ele_seed_dphi2posVsPt_->Fill(bestGsfElectron.pt(), elseed->dPhiPos(1));
        }
        if (elseed->dRZNeg(1) != std::numeric_limits<float>::infinity()) {
          h1_ele_seed_drz2_->Fill(elseed->dRZNeg(1));
          h2_ele_seed_drz2VsEta_->Fill(bestGsfElectron.eta(), elseed->dRZNeg(1));
          h2_ele_seed_drz2VsPt_->Fill(bestGsfElectron.pt(), elseed->dRZNeg(1));
        }
        if (elseed->dRZPos(1) != std::numeric_limits<float>::infinity()) {
          h1_ele_seed_drz2pos_->Fill(elseed->dRZPos(1));
          h2_ele_seed_drz2posVsEta_->Fill(bestGsfElectron.eta(), elseed->dRZPos(1));
          h2_ele_seed_drz2posVsPt_->Fill(bestGsfElectron.pt(), elseed->dRZPos(1));
        }
      }
      // match distributions
      h1_ele_EoP->Fill(bestGsfElectron.eSuperClusterOverP());
      if (bestGsfElectron.isEB())
        h1_ele_EoP_barrel->Fill(bestGsfElectron.eSuperClusterOverP());
      if (bestGsfElectron.isEE())
        h1_ele_EoP_endcaps->Fill(bestGsfElectron.eSuperClusterOverP());
      h2_ele_EoPVsEta->Fill(bestGsfElectron.eta(), bestGsfElectron.eSuperClusterOverP());
      h2_ele_EoPVsPhi->Fill(bestGsfElectron.phi(), bestGsfElectron.eSuperClusterOverP());
      h2_ele_EoPVsE->Fill(bestGsfElectron.caloEnergy(), bestGsfElectron.eSuperClusterOverP());
      h1_ele_EseedOP->Fill(bestGsfElectron.eSeedClusterOverP());
      if (bestGsfElectron.isEB())
        h1_ele_EseedOP_barrel->Fill(bestGsfElectron.eSeedClusterOverP());
      if (bestGsfElectron.isEE())
        h1_ele_EseedOP_endcaps->Fill(bestGsfElectron.eSeedClusterOverP());
      h2_ele_EseedOPVsEta->Fill(bestGsfElectron.eta(), bestGsfElectron.eSeedClusterOverP());
      h2_ele_EseedOPVsPhi->Fill(bestGsfElectron.phi(), bestGsfElectron.eSeedClusterOverP());
      h2_ele_EseedOPVsE->Fill(bestGsfElectron.caloEnergy(), bestGsfElectron.eSeedClusterOverP());
      h1_ele_EoPout->Fill(bestGsfElectron.eSeedClusterOverPout());
      if (bestGsfElectron.isEB())
        h1_ele_EoPout_barrel->Fill(bestGsfElectron.eSeedClusterOverPout());
      if (bestGsfElectron.isEE())
        h1_ele_EoPout_endcaps->Fill(bestGsfElectron.eSeedClusterOverPout());
      h2_ele_EoPoutVsEta->Fill(bestGsfElectron.eta(), bestGsfElectron.eSeedClusterOverPout());
      h2_ele_EoPoutVsPhi->Fill(bestGsfElectron.phi(), bestGsfElectron.eSeedClusterOverPout());
      h2_ele_EoPoutVsE->Fill(bestGsfElectron.caloEnergy(), bestGsfElectron.eSeedClusterOverPout());
      h1_ele_EeleOPout->Fill(bestGsfElectron.eEleClusterOverPout());
      if (bestGsfElectron.isEB())
        h1_ele_EeleOPout_barrel->Fill(bestGsfElectron.eEleClusterOverPout());
      if (bestGsfElectron.isEE())
        h1_ele_EeleOPout_endcaps->Fill(bestGsfElectron.eEleClusterOverPout());
      h2_ele_EeleOPoutVsEta->Fill(bestGsfElectron.eta(), bestGsfElectron.eEleClusterOverPout());
      h2_ele_EeleOPoutVsPhi->Fill(bestGsfElectron.phi(), bestGsfElectron.eEleClusterOverPout());
      h2_ele_EeleOPoutVsE->Fill(bestGsfElectron.caloEnergy(), bestGsfElectron.eEleClusterOverPout());
      h1_ele_dEtaSc_propVtx->Fill(bestGsfElectron.deltaEtaSuperClusterTrackAtVtx());
      if (bestGsfElectron.isEB())
        h1_ele_dEtaSc_propVtx_barrel->Fill(bestGsfElectron.deltaEtaSuperClusterTrackAtVtx());
      if (bestGsfElectron.isEE())
        h1_ele_dEtaSc_propVtx_endcaps->Fill(bestGsfElectron.deltaEtaSuperClusterTrackAtVtx());
      h2_ele_dEtaScVsEta_propVtx->Fill(bestGsfElectron.eta(), bestGsfElectron.deltaEtaSuperClusterTrackAtVtx());
      h2_ele_dEtaScVsPhi_propVtx->Fill(bestGsfElectron.phi(), bestGsfElectron.deltaEtaSuperClusterTrackAtVtx());
      h2_ele_dEtaScVsPt_propVtx->Fill(bestGsfElectron.pt(), bestGsfElectron.deltaEtaSuperClusterTrackAtVtx());
      h1_ele_dPhiSc_propVtx->Fill(bestGsfElectron.deltaPhiSuperClusterTrackAtVtx());
      if (bestGsfElectron.isEB())
        h1_ele_dPhiSc_propVtx_barrel->Fill(bestGsfElectron.deltaPhiSuperClusterTrackAtVtx());
      if (bestGsfElectron.isEE())
        h1_ele_dPhiSc_propVtx_endcaps->Fill(bestGsfElectron.deltaPhiSuperClusterTrackAtVtx());
      h2_ele_dPhiScVsEta_propVtx->Fill(bestGsfElectron.eta(), bestGsfElectron.deltaPhiSuperClusterTrackAtVtx());
      h2_ele_dPhiScVsPhi_propVtx->Fill(bestGsfElectron.phi(), bestGsfElectron.deltaPhiSuperClusterTrackAtVtx());
      h2_ele_dPhiScVsPt_propVtx->Fill(bestGsfElectron.pt(), bestGsfElectron.deltaPhiSuperClusterTrackAtVtx());
      h1_ele_dEtaCl_propOut->Fill(bestGsfElectron.deltaEtaSeedClusterTrackAtCalo());
      if (bestGsfElectron.isEB())
        h1_ele_dEtaCl_propOut_barrel->Fill(bestGsfElectron.deltaEtaSeedClusterTrackAtCalo());
      if (bestGsfElectron.isEE())
        h1_ele_dEtaCl_propOut_endcaps->Fill(bestGsfElectron.deltaEtaSeedClusterTrackAtCalo());
      h2_ele_dEtaClVsEta_propOut->Fill(bestGsfElectron.eta(), bestGsfElectron.deltaEtaSeedClusterTrackAtCalo());
      h2_ele_dEtaClVsPhi_propOut->Fill(bestGsfElectron.phi(), bestGsfElectron.deltaEtaSeedClusterTrackAtCalo());
      h2_ele_dEtaClVsPt_propOut->Fill(bestGsfElectron.pt(), bestGsfElectron.deltaEtaSeedClusterTrackAtCalo());
      h1_ele_dPhiCl_propOut->Fill(bestGsfElectron.deltaPhiSeedClusterTrackAtCalo());
      if (bestGsfElectron.isEB())
        h1_ele_dPhiCl_propOut_barrel->Fill(bestGsfElectron.deltaPhiSeedClusterTrackAtCalo());
      if (bestGsfElectron.isEE())
        h1_ele_dPhiCl_propOut_endcaps->Fill(bestGsfElectron.deltaPhiSeedClusterTrackAtCalo());
      h2_ele_dPhiClVsEta_propOut->Fill(bestGsfElectron.eta(), bestGsfElectron.deltaPhiSeedClusterTrackAtCalo());
      h2_ele_dPhiClVsPhi_propOut->Fill(bestGsfElectron.phi(), bestGsfElectron.deltaPhiSeedClusterTrackAtCalo());
      h2_ele_dPhiClVsPt_propOut->Fill(bestGsfElectron.pt(), bestGsfElectron.deltaPhiSeedClusterTrackAtCalo());
      h1_ele_dEtaEleCl_propOut->Fill(bestGsfElectron.deltaEtaEleClusterTrackAtCalo());
      if (bestGsfElectron.isEB())
        h1_ele_dEtaEleCl_propOut_barrel->Fill(bestGsfElectron.deltaEtaEleClusterTrackAtCalo());
      if (bestGsfElectron.isEE())
        h1_ele_dEtaEleCl_propOut_endcaps->Fill(bestGsfElectron.deltaEtaEleClusterTrackAtCalo());
      h2_ele_dEtaEleClVsEta_propOut->Fill(bestGsfElectron.eta(), bestGsfElectron.deltaEtaEleClusterTrackAtCalo());
      h2_ele_dEtaEleClVsPhi_propOut->Fill(bestGsfElectron.phi(), bestGsfElectron.deltaEtaEleClusterTrackAtCalo());
      h2_ele_dEtaEleClVsPt_propOut->Fill(bestGsfElectron.pt(), bestGsfElectron.deltaEtaEleClusterTrackAtCalo());
      h1_ele_dPhiEleCl_propOut->Fill(bestGsfElectron.deltaPhiEleClusterTrackAtCalo());
      if (bestGsfElectron.isEB())
        h1_ele_dPhiEleCl_propOut_barrel->Fill(bestGsfElectron.deltaPhiEleClusterTrackAtCalo());
      if (bestGsfElectron.isEE())
        h1_ele_dPhiEleCl_propOut_endcaps->Fill(bestGsfElectron.deltaPhiEleClusterTrackAtCalo());
      h2_ele_dPhiEleClVsEta_propOut->Fill(bestGsfElectron.eta(), bestGsfElectron.deltaPhiEleClusterTrackAtCalo());
      h2_ele_dPhiEleClVsPhi_propOut->Fill(bestGsfElectron.phi(), bestGsfElectron.deltaPhiEleClusterTrackAtCalo());
      h2_ele_dPhiEleClVsPt_propOut->Fill(bestGsfElectron.pt(), bestGsfElectron.deltaPhiEleClusterTrackAtCalo());
      h1_ele_HoE->Fill(bestGsfElectron.hadronicOverEm());
      h1_ele_HoE_bc->Fill(bestGsfElectron.hcalOverEcalBc());
      if (bestGsfElectron.isEB())
        h1_ele_HoE_bc_barrel->Fill(bestGsfElectron.hcalOverEcalBc());
      if (bestGsfElectron.isEE())
        h1_ele_HoE_bc_endcaps->Fill(bestGsfElectron.hcalOverEcalBc());
      if (bestGsfElectron.isEB())
        h1_ele_HoE_barrel->Fill(bestGsfElectron.hadronicOverEm());
      if (bestGsfElectron.isEE())
        h1_ele_HoE_endcaps->Fill(bestGsfElectron.hadronicOverEm());
      if (!bestGsfElectron.isEBEtaGap() && !bestGsfElectron.isEBPhiGap() && !bestGsfElectron.isEBEEGap() &&
          !bestGsfElectron.isEERingGap() && !bestGsfElectron.isEEDeeGap())
        h1_ele_HoE_fiducial->Fill(bestGsfElectron.hadronicOverEm());
      h2_ele_HoEVsEta->Fill(bestGsfElectron.eta(), bestGsfElectron.hadronicOverEm());
      h2_ele_HoEVsPhi->Fill(bestGsfElectron.phi(), bestGsfElectron.hadronicOverEm());
      h2_ele_HoEVsE->Fill(bestGsfElectron.caloEnergy(), bestGsfElectron.hadronicOverEm());

      //classes
      int eleClass = bestGsfElectron.classification();
      if (bestGsfElectron.isEE())
        eleClass += 10;
      h1_ele_classes->Fill(eleClass);

      h1_ele_eta->Fill(std::abs(bestGsfElectron.eta()));
      if (bestGsfElectron.classification() == GsfElectron::GOLDEN)
        h1_ele_eta_golden->Fill(std::abs(bestGsfElectron.eta()));
      if (bestGsfElectron.classification() == GsfElectron::BIGBREM)
        h1_ele_eta_bbrem->Fill(std::abs(bestGsfElectron.eta()));
      //if (bestGsfElectron.classification() == GsfElectron::OLDNARROW) h1_ele_eta_narrow->Fill(std::abs(bestGsfElectron.eta()));
      if (bestGsfElectron.classification() == GsfElectron::SHOWERING)
        h1_ele_eta_shower->Fill(std::abs(bestGsfElectron.eta()));

      // fbrem

      double fbrem_mode = bestGsfElectron.fbrem();
      h1_ele_fbrem->Fill(fbrem_mode);
      p1_ele_fbremVsEta_mode->Fill(bestGsfElectron.eta(), fbrem_mode);

      if (bestGsfElectron.isEB()) {
        double fbrem_mode_barrel = bestGsfElectron.fbrem();
        h1_ele_fbrem_barrel->Fill(fbrem_mode_barrel);
      }

      if (bestGsfElectron.isEE()) {
        double fbrem_mode_endcaps = bestGsfElectron.fbrem();
        h1_ele_fbrem_endcaps->Fill(fbrem_mode_endcaps);
      }

      // new 2014/02/12
      double superclusterfbrem_mode = bestGsfElectron.superClusterFbrem();
      h1_ele_superclusterfbrem->Fill(superclusterfbrem_mode);

      if (bestGsfElectron.isEB()) {
        double superclusterfbrem_mode_barrel = bestGsfElectron.superClusterFbrem();
        h1_ele_superclusterfbrem_barrel->Fill(superclusterfbrem_mode_barrel);
      }

      if (bestGsfElectron.isEE()) {
        double superclusterfbrem_mode_endcaps = bestGsfElectron.superClusterFbrem();
        h1_ele_superclusterfbrem_endcaps->Fill(superclusterfbrem_mode_endcaps);
      }

      if (!readAOD_)  // track extra does not exist in AOD
      {
        double fbrem_mean =
            1. - bestGsfElectron.gsfTrack()->outerMomentum().R() / bestGsfElectron.gsfTrack()->innerMomentum().R();
        p1_ele_fbremVsEta_mean->Fill(bestGsfElectron.eta(), fbrem_mean);
      }

      //

      if (bestGsfElectron.classification() == GsfElectron::GOLDEN)
        h2_ele_PinVsPoutGolden_mode->Fill(bestGsfElectron.trackMomentumOut().R(),
                                          bestGsfElectron.trackMomentumAtVtx().R());
      if (bestGsfElectron.classification() == GsfElectron::SHOWERING)
        h2_ele_PinVsPoutShowering_mode->Fill(bestGsfElectron.trackMomentumOut().R(),
                                             bestGsfElectron.trackMomentumAtVtx().R());
      if (!readAOD_)  // track extra does not exist in AOD
        if (bestGsfElectron.classification() == GsfElectron::GOLDEN)
          h2_ele_PinVsPoutGolden_mean->Fill(bestGsfElectron.gsfTrack()->outerMomentum().R(),
                                            bestGsfElectron.gsfTrack()->innerMomentum().R());
      if (!readAOD_)  // track extra does not exist in AOD
        if (bestGsfElectron.classification() == GsfElectron::SHOWERING)
          h2_ele_PinVsPoutShowering_mean->Fill(bestGsfElectron.gsfTrack()->outerMomentum().R(),
                                               bestGsfElectron.gsfTrack()->innerMomentum().R());
      if (bestGsfElectron.classification() == GsfElectron::GOLDEN)
        h2_ele_PtinVsPtoutGolden_mode->Fill(bestGsfElectron.trackMomentumOut().Rho(),
                                            bestGsfElectron.trackMomentumAtVtx().Rho());
      if (bestGsfElectron.classification() == GsfElectron::SHOWERING)
        h2_ele_PtinVsPtoutShowering_mode->Fill(bestGsfElectron.trackMomentumOut().Rho(),
                                               bestGsfElectron.trackMomentumAtVtx().Rho());
      if (!readAOD_)  // track extra does not exist in AOD
        if (bestGsfElectron.classification() == GsfElectron::GOLDEN)
          h2_ele_PtinVsPtoutGolden_mean->Fill(bestGsfElectron.gsfTrack()->outerMomentum().Rho(),
                                              bestGsfElectron.gsfTrack()->innerMomentum().Rho());
      if (!readAOD_)  // track extra does not exist in AOD
        if (bestGsfElectron.classification() == GsfElectron::SHOWERING)
          h2_ele_PtinVsPtoutShowering_mean->Fill(bestGsfElectron.gsfTrack()->outerMomentum().Rho(),
                                                 bestGsfElectron.gsfTrack()->innerMomentum().Rho());

      h1_ele_mva->Fill(bestGsfElectron.mva_e_pi());
      if (bestGsfElectron.isEB())
        h1_ele_mva_barrel->Fill(bestGsfElectron.mva_e_pi());
      if (bestGsfElectron.isEE())
        h1_ele_mva_endcaps->Fill(bestGsfElectron.mva_e_pi());
      h1_ele_mva_isolated->Fill(bestGsfElectron.mva_Isolated());
      if (bestGsfElectron.isEB())
        h1_ele_mva_barrel_isolated->Fill(bestGsfElectron.mva_Isolated());
      if (bestGsfElectron.isEE())
        h1_ele_mva_endcaps_isolated->Fill(bestGsfElectron.mva_Isolated());
      if (bestGsfElectron.ecalDrivenSeed())
        h1_ele_provenance->Fill(1.);
      if (bestGsfElectron.trackerDrivenSeed())
        h1_ele_provenance->Fill(-1.);
      if (bestGsfElectron.trackerDrivenSeed() || bestGsfElectron.ecalDrivenSeed())
        h1_ele_provenance->Fill(0.);
      if (bestGsfElectron.trackerDrivenSeed() && !bestGsfElectron.ecalDrivenSeed())
        h1_ele_provenance->Fill(-2.);
      if (!bestGsfElectron.trackerDrivenSeed() && bestGsfElectron.ecalDrivenSeed())
        h1_ele_provenance->Fill(2.);
      if (bestGsfElectron.ecalDrivenSeed() && bestGsfElectron.isEB())
        h1_ele_provenance_barrel->Fill(1.);
      if (bestGsfElectron.trackerDrivenSeed() && bestGsfElectron.isEB())
        h1_ele_provenance_barrel->Fill(-1.);
      if ((bestGsfElectron.trackerDrivenSeed() || bestGsfElectron.ecalDrivenSeed()) && bestGsfElectron.isEB())
        h1_ele_provenance_barrel->Fill(0.);
      if (bestGsfElectron.trackerDrivenSeed() && !bestGsfElectron.ecalDrivenSeed() && bestGsfElectron.isEB())
        h1_ele_provenance_barrel->Fill(-2.);
      if (!bestGsfElectron.trackerDrivenSeed() && bestGsfElectron.ecalDrivenSeed() && bestGsfElectron.isEB())
        h1_ele_provenance_barrel->Fill(2.);
      if (bestGsfElectron.ecalDrivenSeed() && bestGsfElectron.isEE())
        h1_ele_provenance_endcaps->Fill(1.);
      if (bestGsfElectron.trackerDrivenSeed() && bestGsfElectron.isEE())
        h1_ele_provenance_endcaps->Fill(-1.);
      if ((bestGsfElectron.trackerDrivenSeed() || bestGsfElectron.ecalDrivenSeed()) && bestGsfElectron.isEE())
        h1_ele_provenance_endcaps->Fill(0.);
      if (bestGsfElectron.trackerDrivenSeed() && !bestGsfElectron.ecalDrivenSeed() && bestGsfElectron.isEE())
        h1_ele_provenance_endcaps->Fill(-2.);
      if (!bestGsfElectron.trackerDrivenSeed() && bestGsfElectron.ecalDrivenSeed() && bestGsfElectron.isEE())
        h1_ele_provenance_endcaps->Fill(2.);

      // Pflow isolation
      h1_ele_chargedHadronIso->Fill(bestGsfElectron.pfIsolationVariables().sumChargedHadronPt);
      if (bestGsfElectron.isEB())
        h1_ele_chargedHadronIso_barrel->Fill(bestGsfElectron.pfIsolationVariables().sumChargedHadronPt);
      if (bestGsfElectron.isEE())
        h1_ele_chargedHadronIso_endcaps->Fill(bestGsfElectron.pfIsolationVariables().sumChargedHadronPt);

      h1_ele_neutralHadronIso->Fill(bestGsfElectron.pfIsolationVariables().sumNeutralHadronEt);
      if (bestGsfElectron.isEB())
        h1_ele_neutralHadronIso_barrel->Fill(bestGsfElectron.pfIsolationVariables().sumNeutralHadronEt);
      if (bestGsfElectron.isEE())
        h1_ele_neutralHadronIso_endcaps->Fill(bestGsfElectron.pfIsolationVariables().sumNeutralHadronEt);

      h1_ele_photonIso->Fill(bestGsfElectron.pfIsolationVariables().sumPhotonEt);
      if (bestGsfElectron.isEB())
        h1_ele_photonIso_barrel->Fill(bestGsfElectron.pfIsolationVariables().sumPhotonEt);
      if (bestGsfElectron.isEE())
        h1_ele_photonIso_endcaps->Fill(bestGsfElectron.pfIsolationVariables().sumPhotonEt);

      // -- pflow over pT
      h1_ele_chargedHadronRelativeIso->Fill(bestGsfElectron.pfIsolationVariables().sumChargedHadronPt /
                                            bestGsfElectron.pt());
      if (bestGsfElectron.isEB())
        h1_ele_chargedHadronRelativeIso_barrel->Fill(bestGsfElectron.pfIsolationVariables().sumChargedHadronPt /
                                                     bestGsfElectron.pt());
      if (bestGsfElectron.isEE())
        h1_ele_chargedHadronRelativeIso_endcaps->Fill(bestGsfElectron.pfIsolationVariables().sumChargedHadronPt /
                                                      bestGsfElectron.pt());

      h1_ele_neutralHadronRelativeIso->Fill(bestGsfElectron.pfIsolationVariables().sumNeutralHadronEt /
                                            bestGsfElectron.pt());
      if (bestGsfElectron.isEB())
        h1_ele_neutralHadronRelativeIso_barrel->Fill(bestGsfElectron.pfIsolationVariables().sumNeutralHadronEt /
                                                     bestGsfElectron.pt());
      if (bestGsfElectron.isEE())
        h1_ele_neutralHadronRelativeIso_endcaps->Fill(bestGsfElectron.pfIsolationVariables().sumNeutralHadronEt /
                                                      bestGsfElectron.pt());

      h1_ele_photonRelativeIso->Fill(bestGsfElectron.pfIsolationVariables().sumPhotonEt / bestGsfElectron.pt());
      if (bestGsfElectron.isEB())
        h1_ele_photonRelativeIso_barrel->Fill(bestGsfElectron.pfIsolationVariables().sumPhotonEt /
                                              bestGsfElectron.pt());
      if (bestGsfElectron.isEE())
        h1_ele_photonRelativeIso_endcaps->Fill(bestGsfElectron.pfIsolationVariables().sumPhotonEt /
                                               bestGsfElectron.pt());

      // isolation
      h1_ele_tkSumPt_dr03->Fill(bestGsfElectron.dr03TkSumPt());
      if (bestGsfElectron.isEB())
        h1_ele_tkSumPt_dr03_barrel->Fill(bestGsfElectron.dr03TkSumPt());
      if (bestGsfElectron.isEE())
        h1_ele_tkSumPt_dr03_endcaps->Fill(bestGsfElectron.dr03TkSumPt());
      h1_ele_ecalRecHitSumEt_dr03->Fill(bestGsfElectron.dr03EcalRecHitSumEt());
      if (bestGsfElectron.isEB())
        h1_ele_ecalRecHitSumEt_dr03_barrel->Fill(bestGsfElectron.dr03EcalRecHitSumEt());
      if (bestGsfElectron.isEE())
        h1_ele_ecalRecHitSumEt_dr03_endcaps->Fill(bestGsfElectron.dr03EcalRecHitSumEt());
      h1_ele_hcalTowerSumEt_dr03_depth1->Fill(bestGsfElectron.dr03HcalDepth1TowerSumEt());
      if (bestGsfElectron.isEB())
        h1_ele_hcalTowerSumEt_dr03_depth1_barrel->Fill(bestGsfElectron.dr03HcalDepth1TowerSumEt());
      if (bestGsfElectron.isEE())
        h1_ele_hcalTowerSumEt_dr03_depth1_endcaps->Fill(bestGsfElectron.dr03HcalDepth1TowerSumEt());
      h1_ele_hcalTowerSumEt_dr03_depth2->Fill(bestGsfElectron.dr03HcalDepth2TowerSumEt());
      h1_ele_hcalTowerSumEtBc_dr03_depth1->Fill(bestGsfElectron.dr03HcalDepth1TowerSumEtBc());
      if (bestGsfElectron.isEB())
        h1_ele_hcalTowerSumEtBc_dr03_depth1_barrel->Fill(bestGsfElectron.dr03HcalDepth1TowerSumEtBc());
      if (bestGsfElectron.isEE())
        h1_ele_hcalTowerSumEtBc_dr03_depth1_endcaps->Fill(bestGsfElectron.dr03HcalDepth1TowerSumEtBc());
      h1_ele_hcalTowerSumEtBc_dr03_depth2->Fill(bestGsfElectron.dr03HcalDepth2TowerSumEtBc());
      if (bestGsfElectron.isEB())
        h1_ele_hcalTowerSumEtBc_dr03_depth2_barrel->Fill(bestGsfElectron.dr03HcalDepth2TowerSumEtBc());
      if (bestGsfElectron.isEE())
        h1_ele_hcalTowerSumEtBc_dr03_depth2_endcaps->Fill(bestGsfElectron.dr03HcalDepth2TowerSumEtBc());
      h1_ele_tkSumPt_dr04->Fill(bestGsfElectron.dr04TkSumPt());
      if (bestGsfElectron.isEB())
        h1_ele_tkSumPt_dr04_barrel->Fill(bestGsfElectron.dr04TkSumPt());
      if (bestGsfElectron.isEE())
        h1_ele_tkSumPt_dr04_endcaps->Fill(bestGsfElectron.dr04TkSumPt());
      h1_ele_ecalRecHitSumEt_dr04->Fill(bestGsfElectron.dr04EcalRecHitSumEt());
      if (bestGsfElectron.isEB())
        h1_ele_ecalRecHitSumEt_dr04_barrel->Fill(bestGsfElectron.dr04EcalRecHitSumEt());
      if (bestGsfElectron.isEE())
        h1_ele_ecalRecHitSumEt_dr04_endcaps->Fill(bestGsfElectron.dr04EcalRecHitSumEt());
      h1_ele_hcalTowerSumEt_dr04_depth1->Fill(bestGsfElectron.dr04HcalDepth1TowerSumEt());
      if (bestGsfElectron.isEB())
        h1_ele_hcalTowerSumEt_dr04_depth1_barrel->Fill(bestGsfElectron.dr04HcalDepth1TowerSumEt());
      if (bestGsfElectron.isEE())
        h1_ele_hcalTowerSumEt_dr04_depth1_endcaps->Fill(bestGsfElectron.dr04HcalDepth1TowerSumEt());
      h1_ele_hcalTowerSumEt_dr04_depth2->Fill(bestGsfElectron.dr04HcalDepth2TowerSumEt());
      h1_ele_hcalTowerSumEtBc_dr04_depth1->Fill(bestGsfElectron.dr04HcalDepth1TowerSumEtBc());
      if (bestGsfElectron.isEB())
        h1_ele_hcalTowerSumEtBc_dr04_depth1_barrel->Fill(bestGsfElectron.dr04HcalDepth1TowerSumEtBc());
      if (bestGsfElectron.isEE())
        h1_ele_hcalTowerSumEtBc_dr04_depth1_endcaps->Fill(bestGsfElectron.dr04HcalDepth1TowerSumEtBc());
      h1_ele_hcalTowerSumEtBc_dr04_depth2->Fill(bestGsfElectron.dr04HcalDepth2TowerSumEtBc());
      if (bestGsfElectron.isEB())
        h1_ele_hcalTowerSumEtBc_dr04_depth2_barrel->Fill(bestGsfElectron.dr04HcalDepth2TowerSumEtBc());
      if (bestGsfElectron.isEE())
        h1_ele_hcalTowerSumEtBc_dr04_depth2_endcaps->Fill(bestGsfElectron.dr04HcalDepth2TowerSumEtBc());

      h1_ele_hcalDepth1OverEcalBc->Fill(bestGsfElectron.hcalDepth1OverEcalBc());
      if (bestGsfElectron.isEB())
        h1_ele_hcalDepth1OverEcalBc_barrel->Fill(bestGsfElectron.hcalDepth1OverEcalBc());
      if (bestGsfElectron.isEE())
        h1_ele_hcalDepth1OverEcalBc_endcaps->Fill(bestGsfElectron.hcalDepth1OverEcalBc());
      h1_ele_hcalDepth2OverEcalBc->Fill(bestGsfElectron.hcalDepth2OverEcalBc());
      if (bestGsfElectron.isEB())
        h1_ele_hcalDepth2OverEcalBc_barrel->Fill(bestGsfElectron.hcalDepth2OverEcalBc());
      if (bestGsfElectron.isEE())
        h1_ele_hcalDepth2OverEcalBc_endcaps->Fill(bestGsfElectron.hcalDepth2OverEcalBc());

      // conversion rejection
      int flags = bestGsfElectron.convFlags();
      if (flags == -9999) {
        flags = -1;
      }
      h1_ele_convFlags->Fill(flags);
      if (flags >= 0.) {
        h1_ele_convDist->Fill(bestGsfElectron.convDist());
        h1_ele_convDcot->Fill(bestGsfElectron.convDcot());
        h1_ele_convRadius->Fill(bestGsfElectron.convRadius());
      }

    }  // gsf electron found

  }  // loop overmatching object

  h1_matchingObjectNum->Fill(matchingObjectNum);
}
