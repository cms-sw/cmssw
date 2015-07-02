
// user include files
#include "Validation/RecoEgamma/plugins/ElectronMcSignalValidator.h" 

#include "RecoEgamma/EgammaElectronAlgos/interface/ElectronUtilities.h"

#include "DQMServices/Core/interface/MonitorElement.h"

#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedFwd.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/ValueMap.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
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

ElectronMcSignalValidator::ElectronMcSignalValidator( const edm::ParameterSet & conf )
 : ElectronDqmAnalyzerBase(conf)
 {
   mcTruthCollection_ = consumes<reco::GenParticleCollection> (
       conf.getParameter<edm::InputTag>("mcTruthCollection"));
  electronCollection_      = consumes<reco::GsfElectronCollection> (
      conf.getParameter<edm::InputTag>("electronCollection"));
  electronCoreCollection_  = consumes<reco::GsfElectronCoreCollection> (
      conf.getParameter<edm::InputTag>("electronCoreCollection"));
  electronTrackCollection_ = consumes<reco::GsfTrackCollection> (
      conf.getParameter<edm::InputTag>("electronTrackCollection"));
  electronSeedCollection_  = consumes<reco::ElectronSeedCollection> (
      conf.getParameter<edm::InputTag>("electronSeedCollection"));
  /* new 03/02/2015 */
  offlineVerticesCollection_ = consumes<reco::VertexCollection> (
      conf.getParameter<edm::InputTag>("offlinePrimaryVertices"));
  /* fin new */
  beamSpotTag_ = consumes<reco::BeamSpot> (
      conf.getParameter<edm::InputTag>("beamSpot"));

  readAOD_ = conf.getParameter<bool>("readAOD");

  isoFromDepsTk03Tag_ = consumes<edm::ValueMap<double> > (
      conf.getParameter<edm::InputTag>( "isoFromDepsTk03"));
  isoFromDepsTk04Tag_ = consumes<edm::ValueMap<double> > (
      conf.getParameter<edm::InputTag>( "isoFromDepsTk04"));
  isoFromDepsEcalFull03Tag_ = consumes<edm::ValueMap<double> > (
      conf.getParameter<edm::InputTag>( "isoFromDepsEcalFull03"));
  isoFromDepsEcalFull04Tag_ = consumes<edm::ValueMap<double> > (
      conf.getParameter<edm::InputTag>( "isoFromDepsEcalFull04"));
  isoFromDepsEcalReduced03Tag_ = consumes<edm::ValueMap<double> >(
      conf.getParameter<edm::InputTag>( "isoFromDepsEcalReduced03"));
  isoFromDepsEcalReduced04Tag_ = consumes<edm::ValueMap<double> > (
      conf.getParameter<edm::InputTag>( "isoFromDepsEcalReduced04"));
  isoFromDepsHcal03Tag_ = consumes<edm::ValueMap<double> > (
      conf.getParameter<edm::InputTag>( "isoFromDepsHcal03"));
  isoFromDepsHcal04Tag_ = consumes<edm::ValueMap<double> > (
      conf.getParameter<edm::InputTag>( "isoFromDepsHcal04"));

  maxPt_ = conf.getParameter<double>("MaxPt");
  maxAbsEta_ = conf.getParameter<double>("MaxAbsEta");
  deltaR_ = conf.getParameter<double>("DeltaR");
  matchingIDs_ = conf.getParameter<std::vector<int> >("MatchingID");
  matchingMotherIDs_ = conf.getParameter<std::vector<int> >("MatchingMotherID");
  inputFile_ = conf.getParameter<std::string>("InputFile") ;
  outputFile_ = conf.getParameter<std::string>("OutputFile") ;
  inputInternalPath_ = conf.getParameter<std::string>("InputFolderName") ;
  outputInternalPath_ = conf.getParameter<std::string>("OutputFolderName") ;

  // histos bining and limits

  edm::ParameterSet histosSet = conf.getParameter<edm::ParameterSet>("histosCfg") ;

  xyz_nbin=histosSet.getParameter<int>("Nbinxyz");

  p_nbin=histosSet.getParameter<int>("Nbinp");
  p2D_nbin=histosSet.getParameter<int>("Nbinp2D");
  p_max=histosSet.getParameter<double>("Pmax");

  pt_nbin=histosSet.getParameter<int>("Nbinpt");
  pt2D_nbin=histosSet.getParameter<int>("Nbinpt2D");
  pteff_nbin=histosSet.getParameter<int>("Nbinpteff");
  pt_max=histosSet.getParameter<double>("Ptmax");

  fhits_nbin=histosSet.getParameter<int>("Nbinfhits");
  fhits_max=histosSet.getParameter<double>("Fhitsmax");

  lhits_nbin=histosSet.getParameter<int>("Nbinlhits");
  lhits_max=histosSet.getParameter<double>("Lhitsmax");

  eop_nbin=histosSet.getParameter<int>("Nbineop");
  eop2D_nbin=histosSet.getParameter<int>("Nbineop2D");
  eop_max=histosSet.getParameter<double>("Eopmax");
  eopmaxsht=histosSet.getParameter<double>("Eopmaxsht");

  eta_nbin=histosSet.getParameter<int>("Nbineta");
  eta2D_nbin=histosSet.getParameter<int>("Nbineta2D");
  eta_min=histosSet.getParameter<double>("Etamin");
  eta_max=histosSet.getParameter<double>("Etamax");

  deta_nbin=histosSet.getParameter<int>("Nbindeta");
  deta_min=histosSet.getParameter<double>("Detamin");
  deta_max=histosSet.getParameter<double>("Detamax");

  phi_nbin=histosSet.getParameter<int>("Nbinphi");
  phi2D_nbin=histosSet.getParameter<int>("Nbinphi2D");
  phi_min=histosSet.getParameter<double>("Phimin");
  phi_max=histosSet.getParameter<double>("Phimax");

  detamatch_nbin=histosSet.getParameter<int>("Nbindetamatch");
  detamatch2D_nbin=histosSet.getParameter<int>("Nbindetamatch2D");
  detamatch_min=histosSet.getParameter<double>("Detamatchmin");
  detamatch_max=histosSet.getParameter<double>("Detamatchmax");

  dphi_nbin=histosSet.getParameter<int>("Nbindphi");
  dphi_min=histosSet.getParameter<double>("Dphimin");
  dphi_max=histosSet.getParameter<double>("Dphimax");

  dphimatch_nbin=histosSet.getParameter<int>("Nbindphimatch");
  dphimatch2D_nbin=histosSet.getParameter<int>("Nbindphimatch2D");
  dphimatch_min=histosSet.getParameter<double>("Dphimatchmin");
  dphimatch_max=histosSet.getParameter<double>("Dphimatchmax");

  poptrue_nbin= histosSet.getParameter<int>("Nbinpoptrue");
  poptrue_min=histosSet.getParameter<double>("Poptruemin");
  poptrue_max=histosSet.getParameter<double>("Poptruemax");

  mee_nbin= histosSet.getParameter<int>("Nbinmee");
  mee_min=histosSet.getParameter<double>("Meemin");
  mee_max=histosSet.getParameter<double>("Meemax");

  hoe_nbin= histosSet.getParameter<int>("Nbinhoe");
  hoe_min=histosSet.getParameter<double>("Hoemin");
  hoe_max=histosSet.getParameter<double>("Hoemax");

  error_nbin=histosSet.getParameter<int>("Nbinerror");
  enerror_max=histosSet.getParameter<double>("Energyerrormax");

  set_EfficiencyFlag=histosSet.getParameter<bool>("EfficiencyFlag");
  set_StatOverflowFlag=histosSet.getParameter<bool>("StatOverflowFlag");

  // so to please coverity...
  h1_mcNum = 0 ;
  h1_eleNum = 0 ;
  h1_gamNum = 0 ;

  h1_recEleNum = 0 ;
  h1_recCoreNum = 0 ;
  h1_recTrackNum = 0 ;
  h1_recSeedNum = 0 ;
  h1_recOfflineVertices = 0 ;

  h1_mc_Eta = 0 ;
  h1_mc_AbsEta = 0 ;
  h1_mc_P = 0 ;
  h1_mc_Pt = 0 ;
  h1_mc_Phi = 0 ;
  h1_mc_Z = 0 ;
  h2_mc_PtEta = 0 ;

  h1_mc_Eta_matched = 0 ;
  h1_mc_AbsEta_matched = 0 ;
  h1_mc_Pt_matched = 0 ;
  h1_mc_Phi_matched = 0 ;
  h1_mc_Z_matched = 0 ;
  h2_mc_PtEta_matched = 0 ;

  h1_mc_Eta_matched_qmisid = 0 ;
  h1_mc_AbsEta_matched_qmisid = 0 ;
  h1_mc_Pt_matched_qmisid = 0 ;
  h1_mc_Phi_matched_qmisid = 0 ;
  h1_mc_Z_matched_qmisid = 0 ;

  h1_ele_EoverP_all = 0 ;
  h1_ele_EoverP_all_barrel = 0 ;
  h1_ele_EoverP_all_endcaps = 0 ;
  h1_ele_EseedOP_all = 0 ;
  h1_ele_EseedOP_all_barrel = 0 ;
  h1_ele_EseedOP_all_endcaps = 0 ;
  h1_ele_EoPout_all = 0 ;
  h1_ele_EoPout_all_barrel = 0 ;
  h1_ele_EoPout_all_endcaps = 0 ;
  h1_ele_EeleOPout_all = 0 ;
  h1_ele_EeleOPout_all_barrel = 0 ;
  h1_ele_EeleOPout_all_endcaps = 0 ;
  h1_ele_dEtaSc_propVtx_all = 0 ;
  h1_ele_dEtaSc_propVtx_all_barrel = 0 ;
  h1_ele_dEtaSc_propVtx_all_endcaps = 0 ;
  h1_ele_dPhiSc_propVtx_all = 0 ;
  h1_ele_dPhiSc_propVtx_all_barrel = 0 ;
  h1_ele_dPhiSc_propVtx_all_endcaps = 0 ;
  h1_ele_dEtaCl_propOut_all = 0 ;
  h1_ele_dEtaCl_propOut_all_barrel = 0 ;
  h1_ele_dEtaCl_propOut_all_endcaps = 0 ;
  h1_ele_dPhiCl_propOut_all = 0 ;
  h1_ele_dPhiCl_propOut_all_barrel = 0 ;
  h1_ele_dPhiCl_propOut_all_endcaps = 0 ;
  h1_ele_TIP_all = 0 ;
  h1_ele_TIP_all_barrel = 0 ;
  h1_ele_TIP_all_endcaps = 0 ;
  h1_ele_HoE_all = 0 ;
  h1_ele_HoE_all_barrel = 0 ;
  h1_ele_HoE_all_endcaps = 0 ;
  h1_ele_vertexEta_all = 0 ;
  h1_ele_vertexPt_all = 0 ;
  h1_ele_Et_all = 0 ;
  h1_ele_mee_all = 0 ;
  h1_ele_mee_os = 0 ;
  h1_ele_mee_os_ebeb = 0 ;
  h1_ele_mee_os_ebee = 0 ;
  h1_ele_mee_os_eeee = 0 ;
  h1_ele_mee_os_gg = 0 ;
  h1_ele_mee_os_gb = 0 ;
  h1_ele_mee_os_bb = 0 ;

  h2_ele_E2mnE1vsMee_all = 0 ;
  h2_ele_E2mnE1vsMee_egeg_all = 0 ;

  h1_ele_charge = 0 ;
  h2_ele_chargeVsEta = 0 ;
  h2_ele_chargeVsPhi = 0 ;
  h2_ele_chargeVsPt = 0 ;
  h1_ele_vertexP = 0 ;
  h1_ele_vertexPt = 0 ;
  h1_ele_Et = 0 ;
  h2_ele_vertexPtVsEta = 0 ;
  h2_ele_vertexPtVsPhi = 0 ;
  h1_ele_vertexPt_5100 = 0 ;
  h1_ele_vertexEta = 0 ;
  h2_ele_vertexEtaVsPhi = 0 ;
  h1_ele_vertexAbsEta = 0 ;
  h1_ele_vertexPhi = 0 ;
  h1_ele_vertexX = 0 ;
  h1_ele_vertexY = 0 ;
  h1_ele_vertexZ = 0 ;
  h1_ele_vertexTIP = 0 ;
  h2_ele_vertexTIPVsEta = 0 ;
  h2_ele_vertexTIPVsPhi = 0 ;
  h2_ele_vertexTIPVsPt = 0 ;

  h1_scl_En = 0 ;
  h1_scl_EoEtrue_barrel = 0 ;
  h1_scl_EoEtrue_endcaps = 0 ;
  h1_scl_EoEtrue_barrel_etagap = 0 ;
  h1_scl_EoEtrue_barrel_phigap = 0 ;
  h1_scl_EoEtrue_ebeegap = 0 ;
  h1_scl_EoEtrue_endcaps_deegap = 0 ;
  h1_scl_EoEtrue_endcaps_ringgap = 0 ;
  h1_scl_EoEtrue_barrel_new = 0 ;
  h1_scl_EoEtrue_endcaps_new = 0 ;
  h1_scl_EoEtrue_barrel_new_etagap = 0 ;
  h1_scl_EoEtrue_barrel_new_phigap = 0 ;
  h1_scl_EoEtrue_ebeegap_new = 0 ;
  h1_scl_EoEtrue_endcaps_new_deegap = 0 ;
  h2_scl_EoEtrueVsrecOfflineVertices = 0 ; // new 2015.15.05
  h2_scl_EoEtrueVsrecOfflineVertices_barrel = 0 ; // new 2015.15.05
  h2_scl_EoEtrueVsrecOfflineVertices_endcaps = 0 ; // new 2015.15.05
  h1_scl_EoEtrue_endcaps_new_ringgap = 0 ;
  h1_scl_Et = 0 ;
  h2_scl_EtVsEta = 0 ;
  h2_scl_EtVsPhi = 0 ;
  h2_scl_EtaVsPhi = 0 ;
  h1_scl_Eta = 0 ;
  h1_scl_Phi = 0 ;

  h2_scl_EoEtruePfVsEg  = 0 ;

  h1_scl_SigEtaEta = 0 ;
  h1_scl_SigEtaEta_barrel = 0 ;
  h1_scl_SigEtaEta_endcaps = 0 ;
  h1_scl_SigIEtaIEta = 0 ;
  h1_scl_SigIEtaIEta_barrel = 0 ;
  h1_scl_SigIEtaIEta_endcaps = 0 ;
  h1_scl_full5x5_sigmaIetaIeta = 0 ;
  h1_scl_full5x5_sigmaIetaIeta_barrel = 0 ;
  h1_scl_full5x5_sigmaIetaIeta_endcaps = 0 ;
  h1_scl_E1x5 = 0 ;
  h1_scl_E1x5_barrel = 0 ;
  h1_scl_E1x5_endcaps = 0 ;
  h1_scl_E2x5max = 0 ;
  h1_scl_E2x5max_barrel = 0 ;
  h1_scl_E2x5max_endcaps = 0 ;
  h1_scl_E5x5 = 0 ;
  h1_scl_E5x5_barrel = 0 ;
  h1_scl_E5x5_endcaps = 0 ;
  h1_scl_bcl_EtotoEtrue = 0 ; // new 2015.18.05
  h1_scl_bcl_EtotoEtrue_barrel = 0 ; // new 2015.18.05
  h1_scl_bcl_EtotoEtrue_endcaps = 0 ; // new 2015.18.05

  h1_ele_ambiguousTracks = 0 ;
  h2_ele_ambiguousTracksVsEta = 0 ;
  h2_ele_ambiguousTracksVsPhi = 0 ;
  h2_ele_ambiguousTracksVsPt = 0 ;
  h1_ele_foundHits = 0 ;
  h1_ele_foundHits_barrel = 0 ;
  h1_ele_foundHits_endcaps = 0 ;
  h2_ele_foundHitsVsEta = 0 ;
  h2_ele_foundHitsVsPhi = 0 ;
  h2_ele_foundHitsVsPt = 0 ;
  h1_ele_lostHits = 0 ;
  h1_ele_lostHits_barrel = 0 ;
  h1_ele_lostHits_endcaps = 0 ;
  h2_ele_lostHitsVsEta = 0 ;
  h2_ele_lostHitsVsPhi = 0 ;
  h2_ele_lostHitsVsPt = 0 ;
  h1_ele_chi2 = 0 ;
  h1_ele_chi2_barrel = 0 ;
  h1_ele_chi2_endcaps = 0 ;
  h2_ele_chi2VsEta = 0 ;
  h2_ele_chi2VsPhi = 0 ;
  h2_ele_chi2VsPt = 0 ;

  h1_ele_PoPtrue = 0 ;
  h1_ele_PoPtrue_barrel = 0 ;
  h1_ele_PoPtrue_endcaps = 0 ;

  h2_ele_PoPtrueVsEta = 0 ;
  h2_ele_PoPtrueVsPhi = 0 ;
  h2_ele_PoPtrueVsPt = 0 ;

  h1_ele_PoPtrue_golden_barrel = 0 ;
  h1_ele_PoPtrue_golden_endcaps = 0 ;
  h1_ele_PoPtrue_showering_barrel = 0 ;
  h1_ele_PoPtrue_showering_endcaps = 0 ;
  h1_ele_PtoPttrue = 0 ;
  h1_ele_PtoPttrue_barrel = 0 ;
  h1_ele_PtoPttrue_endcaps = 0 ;
  h1_ele_ChargeMnChargeTrue = 0 ;
  h1_ele_EtaMnEtaTrue = 0 ;
  h1_ele_EtaMnEtaTrue_barrel = 0 ;
  h1_ele_EtaMnEtaTrue_endcaps = 0 ;
  h2_ele_EtaMnEtaTrueVsEta = 0 ;
  h2_ele_EtaMnEtaTrueVsPhi = 0 ;
  h2_ele_EtaMnEtaTrueVsPt = 0 ;
  h1_ele_PhiMnPhiTrue = 0 ;
  h1_ele_PhiMnPhiTrue_barrel = 0 ;
  h1_ele_PhiMnPhiTrue_endcaps = 0 ;
  h1_ele_PhiMnPhiTrue2 = 0 ;
  h2_ele_PhiMnPhiTrueVsEta = 0 ;
  h2_ele_PhiMnPhiTrueVsPhi = 0 ;
  h2_ele_PhiMnPhiTrueVsPt = 0 ;
  h1_ele_PinMnPout = 0 ;
  h1_ele_PinMnPout_mode = 0 ;
  h2_ele_PinMnPoutVsEta_mode = 0 ;
  h2_ele_PinMnPoutVsPhi_mode = 0 ;
  h2_ele_PinMnPoutVsPt_mode = 0 ;
  h2_ele_PinMnPoutVsE_mode = 0 ;
  h2_ele_PinMnPoutVsChi2_mode = 0 ;

  h1_ele_outerP = 0 ;
  h1_ele_outerP_mode = 0 ;
  h2_ele_outerPVsEta_mode = 0 ;
  h1_ele_outerPt = 0 ;
  h1_ele_outerPt_mode = 0 ;
  h2_ele_outerPtVsEta_mode = 0 ;
  h2_ele_outerPtVsPhi_mode = 0 ;
  h2_ele_outerPtVsPt_mode = 0 ;
  h1_ele_EoP = 0 ;
  h1_ele_EoP_barrel = 0 ;
  h1_ele_EoP_endcaps = 0 ;
  h2_ele_EoPVsEta = 0 ;
  h2_ele_EoPVsPhi = 0 ;
  h2_ele_EoPVsE = 0 ;
  h1_ele_EseedOP = 0 ;
  h1_ele_EseedOP_barrel = 0 ;
  h1_ele_EseedOP_endcaps = 0 ;
  h2_ele_EseedOPVsEta = 0 ;
  h2_ele_EseedOPVsPhi = 0 ;
  h2_ele_EseedOPVsE = 0 ;
  h1_ele_EoPout = 0 ;
  h1_ele_EoPout_barrel = 0 ;
  h1_ele_EoPout_endcaps = 0 ;
  h2_ele_EoPoutVsEta = 0 ;
  h2_ele_EoPoutVsPhi = 0 ;
  h2_ele_EoPoutVsE = 0 ;
  h1_ele_EeleOPout = 0 ;
  h1_ele_EeleOPout_barrel = 0 ;
  h1_ele_EeleOPout_endcaps = 0 ;
  h2_ele_EeleOPoutVsEta = 0 ;
  h2_ele_EeleOPoutVsPhi = 0 ;
  h2_ele_EeleOPoutVsE = 0 ;

  h1_ele_dEtaSc_propVtx = 0 ;
  h1_ele_dEtaSc_propVtx_barrel = 0 ;
  h1_ele_dEtaSc_propVtx_endcaps = 0 ;
  h2_ele_dEtaScVsEta_propVtx = 0 ;
  h2_ele_dEtaScVsPhi_propVtx = 0 ;
  h2_ele_dEtaScVsPt_propVtx = 0 ;
  h1_ele_dPhiSc_propVtx = 0 ;
  h1_ele_dPhiSc_propVtx_barrel = 0 ;
  h1_ele_dPhiSc_propVtx_endcaps = 0 ;
  h2_ele_dPhiScVsEta_propVtx = 0 ;
  h2_ele_dPhiScVsPhi_propVtx = 0 ;
  h2_ele_dPhiScVsPt_propVtx = 0 ;
  h1_ele_dEtaCl_propOut = 0 ;
  h1_ele_dEtaCl_propOut_barrel = 0 ;
  h1_ele_dEtaCl_propOut_endcaps = 0 ;
  h2_ele_dEtaClVsEta_propOut = 0 ;
  h2_ele_dEtaClVsPhi_propOut = 0 ;
  h2_ele_dEtaClVsPt_propOut = 0 ;
  h1_ele_dPhiCl_propOut = 0 ;
  h1_ele_dPhiCl_propOut_barrel = 0 ;
  h1_ele_dPhiCl_propOut_endcaps = 0 ;
  h2_ele_dPhiClVsEta_propOut = 0 ;
  h2_ele_dPhiClVsPhi_propOut = 0 ;
  h2_ele_dPhiClVsPt_propOut = 0 ;
  h1_ele_dEtaEleCl_propOut = 0 ;
  h1_ele_dEtaEleCl_propOut_barrel = 0 ;
  h1_ele_dEtaEleCl_propOut_endcaps = 0 ;
  h2_ele_dEtaEleClVsEta_propOut = 0 ;
  h2_ele_dEtaEleClVsPhi_propOut = 0 ;
  h2_ele_dEtaEleClVsPt_propOut = 0 ;
  h1_ele_dPhiEleCl_propOut = 0 ;
  h1_ele_dPhiEleCl_propOut_barrel = 0 ;
  h1_ele_dPhiEleCl_propOut_endcaps = 0 ;
  h2_ele_dPhiEleClVsEta_propOut = 0 ;
  h2_ele_dPhiEleClVsPhi_propOut = 0 ;
  h2_ele_dPhiEleClVsPt_propOut = 0 ;

  h1_ele_seed_subdet2 = 0 ;
  h1_ele_seed_mask = 0 ;
  h1_ele_seed_mask_bpix = 0 ;
  h1_ele_seed_mask_fpix = 0 ;
  h1_ele_seed_mask_tec = 0 ;
  h1_ele_seed_dphi2 = 0 ;
  h2_ele_seed_dphi2VsEta = 0 ;
  h2_ele_seed_dphi2VsPt = 0 ;
  h1_ele_seed_dphi2pos = 0 ;
  h2_ele_seed_dphi2posVsEta = 0 ;
  h2_ele_seed_dphi2posVsPt = 0 ;
  h1_ele_seed_drz2 = 0 ;
  h2_ele_seed_drz2VsEta = 0 ;
  h2_ele_seed_drz2VsPt = 0 ;
  h1_ele_seed_drz2pos = 0 ;
  h2_ele_seed_drz2posVsEta = 0 ;
  h2_ele_seed_drz2posVsPt = 0 ;

  h1_ele_classes = 0 ;
  h1_ele_eta = 0 ;
  h1_ele_eta_golden = 0 ;
  h1_ele_eta_bbrem = 0 ;
  h1_ele_eta_shower = 0 ;

  h1_ele_HoE = 0 ;
  h1_ele_HoE_barrel = 0 ;
  h1_ele_HoE_endcaps = 0 ;
  h1_ele_HoE_fiducial = 0 ;
  h2_ele_HoEVsEta = 0 ;
  h2_ele_HoEVsPhi = 0 ;
  h2_ele_HoEVsE = 0 ;

  h1_ele_fbrem = 0 ;
  p1_ele_fbremVsEta_mode = 0 ;
  p1_ele_fbremVsEta_mean = 0 ;
  h1_ele_superclusterfbrem = 0 ;
  h1_ele_superclusterfbrem_barrel = 0 ;
  h1_ele_superclusterfbrem_endcaps = 0 ;
  h2_ele_PinVsPoutGolden_mode = 0 ;
  h2_ele_PinVsPoutShowering_mode = 0 ;
  h2_ele_PinVsPoutGolden_mean = 0 ;
  h2_ele_PinVsPoutShowering_mean = 0 ;
  h2_ele_PtinVsPtoutGolden_mode = 0 ;
  h2_ele_PtinVsPtoutShowering_mode = 0 ;
  h2_ele_PtinVsPtoutGolden_mean = 0 ;
  h2_ele_PtinVsPtoutShowering_mean = 0 ;
  h1_scl_EoEtrueGolden_barrel = 0 ;
  h1_scl_EoEtrueGolden_endcaps = 0 ;
  h1_scl_EoEtrueShowering_barrel = 0 ;
  h1_scl_EoEtrueShowering_endcaps = 0 ;

  h1_ele_mva = 0 ;
  h1_ele_mva_isolated = 0;
  h1_ele_provenance = 0 ;

  // isolation
  h1_ele_tkSumPt_dr03 = 0 ;
  h1_ele_tkSumPt_dr03_barrel = 0 ;
  h1_ele_tkSumPt_dr03_endcaps = 0 ;
  h1_ele_ecalRecHitSumEt_dr03 = 0 ;
  h1_ele_ecalRecHitSumEt_dr03_barrel = 0 ;
  h1_ele_ecalRecHitSumEt_dr03_endcaps = 0 ;
  h1_ele_hcalTowerSumEt_dr03_depth1 = 0 ;
  h1_ele_hcalTowerSumEt_dr03_depth1_barrel = 0 ;
  h1_ele_hcalTowerSumEt_dr03_depth1_endcaps = 0 ;
  h1_ele_hcalTowerSumEt_dr03_depth2 = 0 ;
  h1_ele_tkSumPt_dr04 = 0 ;
  h1_ele_tkSumPt_dr04_barrel = 0 ;
  h1_ele_tkSumPt_dr04_endcaps = 0 ;
  h1_ele_ecalRecHitSumEt_dr04 = 0 ;
  h1_ele_ecalRecHitSumEt_dr04_barrel = 0 ;
  h1_ele_ecalRecHitSumEt_dr04_endcaps = 0 ;
  h1_ele_hcalTowerSumEt_dr04_depth1 = 0 ;
  h1_ele_hcalTowerSumEt_dr04_depth1_barrel = 0 ;
  h1_ele_hcalTowerSumEt_dr04_depth1_endcaps = 0 ;
  h1_ele_hcalTowerSumEt_dr04_depth2 = 0 ;

  // conversions
  h1_ele_convFlags = 0 ;
  h1_ele_convFlags_all = 0 ;
  h1_ele_convDist = 0 ;
  h1_ele_convDist_all = 0 ;
  h1_ele_convDcot = 0 ;
  h1_ele_convDcot_all = 0 ;
  h1_ele_convRadius = 0 ;
  h1_ele_convRadius_all = 0 ;
 }

void ElectronMcSignalValidator::bookHistograms( DQMStore::IBooker & iBooker, edm::Run const &, edm::EventSetup const & )
 {
  iBooker.setCurrentFolder(outputInternalPath_) ;

  //  prepareStore() ;
  setBookIndex(-1) ;
  setBookPrefix("h") ;
  setBookEfficiencyFlag(set_EfficiencyFlag);
  setBookStatOverflowFlag( set_StatOverflowFlag ) ;

  // mc truth collections sizes
  h1_mcNum = bookH1withSumw2(iBooker, "mcNum","# mc particles",fhits_nbin,0.,fhits_max,"N_{gen}" );
  h1_eleNum = bookH1withSumw2(iBooker, "mcNum_ele","# mc electrons",fhits_nbin,0.,fhits_max,"N_{gen ele}");
  h1_gamNum = bookH1withSumw2(iBooker, "mcNum_gam","# mc gammas",fhits_nbin,0.,fhits_max,"N_{gen #gamma}");

  // rec event collections sizes
  h1_recEleNum = bookH1(iBooker, "recEleNum","# rec electrons",11, -0.5,10.5,"N_{ele}");
  h1_recCoreNum = bookH1(iBooker, "recCoreNum","# rec electron cores",21, -0.5,20.5,"N_{core}");
  h1_recTrackNum = bookH1(iBooker, "recTrackNum","# rec gsf tracks",41, -0.5,40.5,"N_{track}");
  h1_recSeedNum = bookH1(iBooker, "recSeedNum","# rec electron seeds",101, -0.5,100.5,"N_{seed}");
  h1_recOfflineVertices = bookH1(iBooker, "recOfflineVertices","# rec Offline Primary Vertices",61, -0.5,60.5,"N_{Vertices}"); 
  h2_scl_EoEtrueVsrecOfflineVertices = bookH2(iBooker, "scl_EoEtrueVsrecOfflineVertices", "E/Etrue vs number of primary vertices", 10, 0., 50., 50, 0., 2.5, "N_{primary vertices}", "E/E_{true}");
  h2_scl_EoEtrueVsrecOfflineVertices_barrel = bookH2(iBooker, "scl_EoEtrueVsrecOfflineVertices_barrel", "E/Etrue vs number of primary , barrel", 10, 0., 50., 50, 0., 2.5, "N_{primary vertices}", "E/E_{true}");
  h2_scl_EoEtrueVsrecOfflineVertices_endcaps = bookH2(iBooker, "scl_EoEtrueVsrecOfflineVertices_endcaps", "E/Etrue vs number of primary , endcaps", 10, 0., 50., 50, 0., 2.5, "N_{primary vertices}", "E/E_{true}");

  // mc
  setBookPrefix("h_mc") ;
  h1_mc_Eta = bookH1withSumw2(iBooker, "Eta","gen #eta",eta_nbin,eta_min,eta_max,"#eta");
  h1_mc_AbsEta = bookH1withSumw2(iBooker, "AbsEta","gen |#eta|",eta_nbin/2,0.,eta_max);
  h1_mc_P = bookH1withSumw2(iBooker, "P","gen p",p_nbin,0.,p_max,"p (GeV/c)");
  h1_mc_Pt = bookH1withSumw2(iBooker, "Pt","gen pt",pteff_nbin,5.,pt_max);
  h1_mc_Phi = bookH1withSumw2(iBooker, "Phi","gen phi",phi_nbin,phi_min,phi_max);
  h1_mc_Z = bookH1withSumw2(iBooker, "Z","gen z ",xyz_nbin, -25, 25 );
  h2_mc_PtEta = bookH2withSumw2(iBooker, "PtEta","gen pt vs #eta",eta2D_nbin,eta_min,eta_max,pt2D_nbin,5.,pt_max );

  // all electrons
  setBookPrefix("h_ele") ;
  h1_ele_EoverP_all = bookH1withSumw2(iBooker, "EoverP_all","ele E/P_{vertex}, all reco electrons",eop_nbin,0.,eop_max,"E/P_{vertex}","Events","ELE_LOGY E1 P");
  h1_ele_EoverP_all_barrel = bookH1withSumw2(iBooker, "EoverP_all_barrel","ele E/P_{vertex}, all reco electrons, barrel",eop_nbin,0.,eop_max,"E/P_{vertex}","Events","ELE_LOGY E1 P");
  h1_ele_EoverP_all_endcaps = bookH1withSumw2(iBooker, "EoverP_all_endcaps","ele E/P_{vertex}, all reco electrons, endcaps",eop_nbin,0.,eop_max,"E/P_{vertex}","Events","ELE_LOGY E1 P");
  h1_ele_EseedOP_all = bookH1withSumw2(iBooker, "EseedOP_all","ele E_{seed}/P_{vertex}, all reco electrons",eop_nbin,0.,eop_max,"E_{seed}/P_{vertex}","Events","ELE_LOGY E1 P");
  h1_ele_EseedOP_all_barrel = bookH1withSumw2(iBooker, "EseedOP_all_barrel","ele E_{seed}/P_{vertex}, all reco electrons, barrel",eop_nbin,0.,eop_max,"E_{seed}/P_{vertex}","Events","ELE_LOGY E1 P");
  h1_ele_EseedOP_all_endcaps = bookH1withSumw2(iBooker, "EseedOP_all_endcaps","ele E_{seed}/P_{vertex}, all reco electrons, endcaps",eop_nbin,0.,eop_max,"E_{seed}/P_{vertex}","Events","ELE_LOGY E1 P");
  h1_ele_EoPout_all = bookH1withSumw2(iBooker, "EoPout_all","ele E_{seed}/P_{out}, all reco electrons",eop_nbin,0.,eop_max,"E_{seed}/P_{out}","Events","ELE_LOGY E1 P");
  h1_ele_EoPout_all_barrel = bookH1withSumw2(iBooker, "EoPout_all_barrel","ele E_{seed}/P_{out}, all reco electrons barrel",eop_nbin,0.,eop_max,"E_{seed}/P_{out}","Events","ELE_LOGY E1 P");
  h1_ele_EoPout_all_endcaps = bookH1withSumw2(iBooker, "EoPout_all_endcaps","ele E_{seed}/P_{out}, all reco electrons endcaps",eop_nbin,0.,eop_max,"E_{seed}/P_{out}","Events","ELE_LOGY E1 P");
  h1_ele_EeleOPout_all = bookH1withSumw2(iBooker, "EeleOPout_all","ele E_{ele}/P_{out}, all reco electrons",eop_nbin,0.,eop_max,"E_{ele}/P_{out}","Events","ELE_LOGY E1 P");
  h1_ele_EeleOPout_all_barrel = bookH1withSumw2(iBooker, "EeleOPout_all_barrel","ele E_{ele}/P_{out}, all reco electrons barrel",eop_nbin,0.,eop_max,"E_{ele}/P_{out}","Events","ELE_LOGY E1 P");
  h1_ele_EeleOPout_all_endcaps = bookH1withSumw2(iBooker, "EeleOPout_all_endcaps","ele E_{ele}/P_{out}, all reco electrons endcaps",eop_nbin,0.,eop_max,"E_{ele}/P_{out}","Events","ELE_LOGY E1 P");
  h1_ele_dEtaSc_propVtx_all = bookH1withSumw2(iBooker, "dEtaSc_propVtx_all","ele #eta_{sc} - #eta_{tr}, prop from vertex, all reco electrons",detamatch_nbin,detamatch_min,detamatch_max,"#eta_{sc} - #eta_{tr}","Events","ELE_LOGY E1 P");
  h1_ele_dEtaSc_propVtx_all_barrel = bookH1withSumw2(iBooker, "dEtaSc_propVtx_all_barrel","ele #eta_{sc} - #eta_{tr}, prop from vertex, all reco electrons barrel",detamatch_nbin,detamatch_min,detamatch_max,"#eta_{sc} - #eta_{tr}","Events","ELE_LOGY E1 P");
  h1_ele_dEtaSc_propVtx_all_endcaps = bookH1withSumw2(iBooker, "dEtaSc_propVtx_all_endcaps","ele #eta_{sc} - #eta_{tr}, prop from vertex, all reco electrons endcaps",detamatch_nbin,detamatch_min,detamatch_max,"#eta_{sc} - #eta_{tr}","Events","ELE_LOGY E1 P");
  h1_ele_dPhiSc_propVtx_all = bookH1withSumw2(iBooker, "dPhiSc_propVtx_all","ele #phi_{sc} - #phi_{tr}, prop from vertex, all reco electrons",dphimatch_nbin,dphimatch_min,dphimatch_max,"#phi_{sc} - #phi_{tr} (rad)","Events","ELE_LOGY E1 P");
  h1_ele_dPhiSc_propVtx_all_barrel = bookH1withSumw2(iBooker, "dPhiSc_propVtx_all_barrel","ele #phi_{sc} - #phi_{tr}, prop from vertex, all reco electrons barrel",dphimatch_nbin,dphimatch_min,dphimatch_max,"#phi_{sc} - #phi_{tr} (rad)","Events","ELE_LOGY E1 P");
  h1_ele_dPhiSc_propVtx_all_endcaps = bookH1withSumw2(iBooker, "dPhiSc_propVtx_all_endcaps","ele #phi_{sc} - #phi_{tr}, prop from vertex, all reco electrons endcaps",dphimatch_nbin,dphimatch_min,dphimatch_max,"#phi_{sc} - #phi_{tr} (rad)","Events","ELE_LOGY E1 P");
  h1_ele_dEtaCl_propOut_all = bookH1withSumw2(iBooker, "dEtaCl_propOut_all","ele #eta_{cl} - #eta_{tr}, prop from outermost, all reco electrons",detamatch_nbin,detamatch_min,detamatch_max,"#eta_{sc} - #eta_{tr}","Events","ELE_LOGY E1 P");
  h1_ele_dEtaCl_propOut_all_barrel = bookH1withSumw2(iBooker, "dEtaCl_propOut_all_barrel","ele #eta_{cl} - #eta_{tr}, prop from outermost, all reco electrons barrel",detamatch_nbin,detamatch_min,detamatch_max,"#eta_{sc} - #eta_{tr}","Events","ELE_LOGY E1 P");
  h1_ele_dEtaCl_propOut_all_endcaps = bookH1withSumw2(iBooker, "dEtaCl_propOut_all_endcaps","ele #eta_{cl} - #eta_{tr}, prop from outermost, all reco electrons endcaps",detamatch_nbin,detamatch_min,detamatch_max,"#eta_{sc} - #eta_{tr}","Events","ELE_LOGY E1 P");
  h1_ele_dPhiCl_propOut_all = bookH1withSumw2(iBooker, "dPhiCl_propOut_all","ele #phi_{cl} - #phi_{tr}, prop from outermost, all reco electrons",dphimatch_nbin,dphimatch_min,dphimatch_max,"#phi_{sc} - #phi_{tr} (rad)","Events","ELE_LOGY E1 P");
  h1_ele_dPhiCl_propOut_all_barrel = bookH1withSumw2(iBooker, "dPhiCl_propOut_all_barrel","ele #phi_{cl} - #phi_{tr}, prop from outermost, all reco electrons barrel",dphimatch_nbin,dphimatch_min,dphimatch_max,"#phi_{sc} - #phi_{tr} (rad)","Events","ELE_LOGY E1 P");
  h1_ele_dPhiCl_propOut_all_endcaps = bookH1withSumw2(iBooker, "dPhiCl_propOut_all_endcaps","ele #phi_{cl} - #phi_{tr}, prop from outermost, all reco electrons endcaps",dphimatch_nbin,dphimatch_min,dphimatch_max,"#phi_{sc} - #phi_{tr} (rad)","Events","ELE_LOGY E1 P");
  h1_ele_HoE_all = bookH1withSumw2(iBooker, "HoE_all","ele hadronic energy / em energy, all reco electrons",hoe_nbin, hoe_min, hoe_max,"H/E","Events","ELE_LOGY E1 P") ;
  h1_ele_HoE_all_barrel = bookH1withSumw2(iBooker, "HoE_all_barrel","ele hadronic energy / em energy, all reco electrons barrel",hoe_nbin, hoe_min, hoe_max,"H/E","Events","ELE_LOGY E1 P") ;
  h1_ele_HoE_all_endcaps = bookH1withSumw2(iBooker, "HoE_all_endcaps","ele hadronic energy / em energy, all reco electrons endcaps",hoe_nbin, hoe_min, hoe_max,"H/E","Events","ELE_LOGY E1 P") ;
  h1_ele_HoE_bc_all = bookH1withSumw2(iBooker, "HoE_bc_all","ele hadronic energy / em energy, all reco electrons, behind cluster",hoe_nbin, hoe_min, hoe_max,"H/E","Events","ELE_LOGY E1 P") ;
  h1_ele_vertexPt_all = bookH1withSumw2(iBooker, "vertexPt_all","ele p_{T}, all reco electrons",pteff_nbin,5.,pt_max,"","Events","ELE_LOGY E1 P");
  h1_ele_Et_all = bookH1withSumw2(iBooker, "Et_all","ele ecal E_{T}, all reco electrons",pteff_nbin,5.,pt_max,"E_{T} (GeV)","Events","ELE_LOGY E1 P");
  h1_ele_vertexEta_all = bookH1withSumw2(iBooker, "vertexEta_all","ele eta, all reco electrons",eta_nbin,eta_min,eta_max,"","Events","ELE_LOGY E1 P");
  h1_ele_TIP_all = bookH1withSumw2(iBooker, "TIP_all","ele vertex transverse radius, all reco electrons",  100,0.,0.2,"r_{T} (cm)","Events","ELE_LOGY E1 P");
  h1_ele_TIP_all_barrel = bookH1withSumw2(iBooker, "TIP_all_barrel","ele vertex transverse radius, all reco electrons barrel",  100,0.,0.2,"r_{T} (cm)","Events","ELE_LOGY E1 P");
  h1_ele_TIP_all_endcaps = bookH1withSumw2(iBooker, "TIP_all_endcaps","ele vertex transverse radius, all reco electrons endcaps",  100,0.,0.2,"r_{T} (cm)","Events","ELE_LOGY E1 P");
  h1_ele_mee_all = bookH1withSumw2(iBooker, "mee_all","ele pairs invariant mass, all reco electrons",mee_nbin, mee_min, mee_max,"m_{ee} (GeV/c^{2})","Events","ELE_LOGY E1 P");
  h1_ele_mee_os = bookH1withSumw2(iBooker, "mee_os","ele pairs invariant mass, opp. sign",mee_nbin, mee_min, mee_max,"m_{e^{+}e^{-}} (GeV/c^{2})","Events","ELE_LOGY E1 P");
  h1_ele_mee_os_ebeb = bookH1withSumw2(iBooker, "mee_os_ebeb","ele pairs invariant mass, opp. sign, EB-EB",mee_nbin, mee_min, mee_max,"m_{e^{+}e^{-}} (GeV/c^{2})","Events","ELE_LOGY E1 P");
  h1_ele_mee_os_ebee = bookH1withSumw2(iBooker, "mee_os_ebee","ele pairs invariant mass, opp. sign, EB-EE",mee_nbin, mee_min, mee_max,"m_{e^{+}e^{-}} (GeV/c^{2})","Events","ELE_LOGY E1 P");
  h1_ele_mee_os_eeee = bookH1withSumw2(iBooker, "mee_os_eeee","ele pairs invariant mass, opp. sign, EE-EE",mee_nbin, mee_min, mee_max,"m_{e^{+}e^{-}} (GeV/c^{2})","Events","ELE_LOGY E1 P");
  h1_ele_mee_os_gg = bookH1withSumw2(iBooker, "mee_os_gg","ele pairs invariant mass, opp. sign, good-good",mee_nbin, mee_min, mee_max,"m_{e^{+}e^{-}} (GeV/c^{2})","Events","ELE_LOGY E1 P");
  h1_ele_mee_os_gb = bookH1withSumw2(iBooker, "mee_os_gb","ele pairs invariant mass, opp. sign, good-bad",mee_nbin, mee_min, mee_max,"m_{e^{+}e^{-}} (GeV/c^{2})","Events","ELE_LOGY E1 P");
  h1_ele_mee_os_bb = bookH1withSumw2(iBooker, "mee_os_bb","ele pairs invariant mass, opp. sign, bad-bad",mee_nbin, mee_min, mee_max,"m_{e^{+}e^{-}} (GeV/c^{2})","Events","ELE_LOGY E1 P");

  // duplicates
  h2_ele_E2mnE1vsMee_all = bookH2(iBooker, "E2mnE1vsMee_all","E2 - E1 vs ele pairs invariant mass, all electrons",mee_nbin, mee_min, mee_max, 100, -50., 50.,"m_{e^{+}e^{-}} (GeV/c^{2})","E2 - E1 (GeV)");
  h2_ele_E2mnE1vsMee_egeg_all = bookH2(iBooker, "E2mnE1vsMee_egeg_all","E2 - E1 vs ele pairs invariant mass, ecal driven pairs, all electrons",mee_nbin, mee_min, mee_max, 100, -50., 50.,"m_{e^{+}e^{-}} (GeV/c^{2})","E2 - E1 (GeV)");

  // charge ID
  h1_ele_ChargeMnChargeTrue = bookH1withSumw2(iBooker, "ChargeMnChargeTrue","ele charge - gen charge ",5,-1.,4.,"q_{rec} - q_{gen}");
  setBookPrefix("h_mc") ;
  h1_mc_Eta_matched_qmisid = bookH1withSumw2(iBooker, "Eta_matched_qmisid","charge misid vs gen eta",eta_nbin,eta_min,eta_max);
  h1_mc_AbsEta_matched_qmisid = bookH1withSumw2(iBooker, "AbsEta_matched_qmisid","charge misid vs gen |eta|",eta_nbin/2,0.,eta_max);
  h1_mc_Pt_matched_qmisid = bookH1withSumw2(iBooker, "Pt_matched_qmisid","charge misid vs gen transverse momentum",pteff_nbin,5.,pt_max);
  h1_mc_Phi_matched_qmisid = bookH1withSumw2(iBooker, "Phi_matched_qmisid","charge misid vs gen phi",phi_nbin,phi_min,phi_max);
  h1_mc_Z_matched_qmisid = bookH1withSumw2(iBooker, "Z_matched_qmisid","charge misid vs gen z",xyz_nbin, -25, 25 );

  // matched electrons
  setBookPrefix("h_mc") ;
  h1_mc_Eta_matched = bookH1withSumw2(iBooker, "Eta_matched","Efficiency vs gen eta",eta_nbin,eta_min,eta_max);
  h1_mc_AbsEta_matched = bookH1withSumw2(iBooker, "AbsEta_matched","Efficiency vs gen |eta|",eta_nbin/2,0.,2.5);
  h1_mc_Pt_matched = bookH1(iBooker, "Pt_matched","Efficiency vs gen transverse momentum",pteff_nbin,5.,pt_max);
  h1_mc_Phi_matched = bookH1withSumw2(iBooker, "Phi_matched","Efficiency vs gen phi",phi_nbin,phi_min,phi_max);
  h1_mc_Z_matched = bookH1withSumw2(iBooker, "Z_matched","Efficiency vs gen vertex z",xyz_nbin,-25,25);
  h2_mc_PtEta_matched = bookH2withSumw2(iBooker, "PtEta_matched","Efficiency vs pt #eta",eta2D_nbin,eta_min,eta_max,pt2D_nbin,5.,pt_max );
  setBookPrefix("h_ele") ;
  h1_ele_charge = bookH1withSumw2(iBooker, "charge","ele charge",5,-2.5,2.5,"charge");
  h2_ele_chargeVsEta = bookH2(iBooker, "chargeVsEta","ele charge vs eta",eta2D_nbin,eta_min,eta_max,5,-2.,2.);
  h2_ele_chargeVsPhi = bookH2(iBooker, "chargeVsPhi","ele charge vs phi",phi2D_nbin,phi_min,phi_max,5,-2.,2.);
  h2_ele_chargeVsPt = bookH2(iBooker, "chargeVsPt","ele charge vs pt",pt_nbin,0.,100.,5,-2.,2.);
  h1_ele_vertexP = bookH1withSumw2(iBooker, "vertexP","ele momentum",p_nbin,0.,p_max,"p_{vertex} (GeV/c)");
  h1_ele_vertexPt = bookH1withSumw2(iBooker, "vertexPt","ele transverse momentum",pt_nbin,0.,pt_max,"p_{T vertex} (GeV/c)");
  h1_ele_Et = bookH1withSumw2(iBooker, "Et","ele ecal E_{T}",pt_nbin,0.,pt_max,"E_{T} (GeV)");
  h2_ele_vertexPtVsEta = bookH2(iBooker, "vertexPtVsEta","ele transverse momentum vs eta",eta2D_nbin,eta_min,eta_max,pt2D_nbin,0.,pt_max);
  h2_ele_vertexPtVsPhi = bookH2(iBooker, "vertexPtVsPhi","ele transverse momentum vs phi",phi2D_nbin,phi_min,phi_max,pt2D_nbin,0.,pt_max);
  h1_ele_vertexEta = bookH1withSumw2(iBooker, "vertexEta","ele momentum eta",eta_nbin,eta_min,eta_max,"#eta");
  h2_ele_vertexEtaVsPhi = bookH2(iBooker, "vertexEtaVsPhi","ele momentum eta vs phi",eta2D_nbin,eta_min,eta_max,phi2D_nbin,phi_min,phi_max );
  h1_ele_vertexPhi = bookH1withSumw2(iBooker, "vertexPhi","ele  momentum #phi",phi_nbin,phi_min,phi_max,"#phi (rad)");
  h1_ele_vertexX = bookH1withSumw2(iBooker, "vertexX","ele vertex x",xyz_nbin,-0.6,0.6,"x (cm)" );
  h1_ele_vertexY = bookH1withSumw2(iBooker, "vertexY","ele vertex y",xyz_nbin,-0.6,0.6,"y (cm)" );
  h1_ele_vertexZ = bookH1withSumw2(iBooker, "vertexZ","ele vertex z",xyz_nbin,-25, 25,"z (cm)" );
  h1_ele_vertexTIP = bookH1withSumw2(iBooker, "vertexTIP","ele transverse impact parameter (wrt gen vtx)",90,0.,0.15,"TIP (cm)");
  h2_ele_vertexTIPVsEta = bookH2(iBooker, "vertexTIPVsEta","ele transverse impact parameter (wrt gen vtx) vs eta",eta2D_nbin,eta_min,eta_max,45,0.,0.15,"#eta","TIP (cm)");
  h2_ele_vertexTIPVsPhi = bookH2(iBooker, "vertexTIPVsPhi","ele transverse impact parameter (wrt gen vtx) vs phi",phi2D_nbin,phi_min,phi_max,45,0.,0.15,"#phi (rad)","TIP (cm)");
  h2_ele_vertexTIPVsPt = bookH2(iBooker, "vertexTIPVsPt","ele transverse impact parameter (wrt gen vtx) vs transverse momentum",pt2D_nbin,0.,pt_max,45,0.,0.15,"p_{T} (GeV/c)","TIP (cm)");
  h1_ele_PoPtrue = bookH1withSumw2(iBooker, "PoPtrue","ele momentum / gen momentum",poptrue_nbin,poptrue_min,poptrue_max,"P/P_{gen}");
  h1_ele_PoPtrue_barrel = bookH1withSumw2(iBooker, "PoPtrue_barrel","ele momentum / gen momentum, barrel",poptrue_nbin,poptrue_min,poptrue_max,"P/P_{gen}");
  h1_ele_PoPtrue_endcaps = bookH1withSumw2(iBooker, "PoPtrue_endcaps","ele momentum / gen momentum, endcaps",poptrue_nbin,poptrue_min,poptrue_max,"P/P_{gen}");
  h2_ele_PoPtrueVsEta = bookH2withSumw2(iBooker, "PoPtrueVsEta","ele momentum / gen momentum vs eta",eta2D_nbin,eta_min,eta_max,50,poptrue_min,poptrue_max);
  h2_ele_PoPtrueVsPhi = bookH2(iBooker, "PoPtrueVsPhi","ele momentum / gen momentum vs phi",phi2D_nbin,phi_min,phi_max,50,poptrue_min,poptrue_max);
  h2_ele_PoPtrueVsPt = bookH2(iBooker, "PoPtrueVsPt","ele momentum / gen momentum vs eta",pt2D_nbin,0.,pt_max,50,poptrue_min,poptrue_max);
  h1_ele_PoPtrue_golden_barrel = bookH1withSumw2(iBooker, "PoPtrue_golden_barrel","ele momentum / gen momentum, golden, barrel",poptrue_nbin,poptrue_min,poptrue_max,"P/P_{gen}");
  h1_ele_PoPtrue_golden_endcaps = bookH1withSumw2(iBooker, "PoPtrue_golden_endcaps","ele momentum / gen momentum, golden, endcaps",poptrue_nbin,poptrue_min,poptrue_max,"P/P_{gen}");
  h1_ele_PoPtrue_showering_barrel = bookH1withSumw2(iBooker, "PoPtrue_showering_barrel","ele momentum / gen momentum, showering, barrel",poptrue_nbin,poptrue_min,poptrue_max,"P/P_{gen}");
  h1_ele_PoPtrue_showering_endcaps = bookH1withSumw2(iBooker, "PoPtrue_showering_endcaps","ele momentum / gen momentum, showering, endcaps",poptrue_nbin,poptrue_min,poptrue_max,"P/P_{gen}");
  h1_ele_PtoPttrue = bookH1withSumw2(iBooker, "PtoPttrue","ele transverse momentum / gen transverse momentum",poptrue_nbin,poptrue_min,poptrue_max,"P_{T}/P_{T}^{gen}");
  h1_ele_PtoPttrue_barrel = bookH1withSumw2(iBooker, "PtoPttrue_barrel","ele transverse momentum / gen transverse momentum, barrel",poptrue_nbin,poptrue_min,poptrue_max,"P_{T}/P_{T}^{gen}");
  h1_ele_PtoPttrue_endcaps = bookH1withSumw2(iBooker, "PtoPttrue_endcaps","ele transverse momentum / gen transverse momentum, endcaps",poptrue_nbin,poptrue_min,poptrue_max,"P_{T}/P_{T}^{gen}");
  h1_ele_EtaMnEtaTrue = bookH1withSumw2(iBooker, "EtaMnEtaTrue","ele momentum  eta - gen  eta",deta_nbin,deta_min,deta_max,"#eta_{rec} - #eta_{gen}");
  h1_ele_EtaMnEtaTrue_barrel = bookH1withSumw2(iBooker, "EtaMnEtaTrue_barrel","ele momentum  eta - gen  eta barrel",deta_nbin,deta_min,deta_max,"#eta_{rec} - #eta_{gen}");
  h1_ele_EtaMnEtaTrue_endcaps = bookH1withSumw2(iBooker, "EtaMnEtaTrue_endcaps","ele momentum  eta - gen  eta endcaps",deta_nbin,deta_min,deta_max,"#eta_{rec} - #eta_{gen}");
  h2_ele_EtaMnEtaTrueVsEta = bookH2(iBooker, "EtaMnEtaTrueVsEta","ele momentum  eta - gen  eta vs eta",eta2D_nbin,eta_min,eta_max,deta_nbin/2,deta_min,deta_max);
  h2_ele_EtaMnEtaTrueVsPhi = bookH2(iBooker, "EtaMnEtaTrueVsPhi","ele momentum  eta - gen  eta vs phi",phi2D_nbin,phi_min,phi_max,deta_nbin/2,deta_min,deta_max);
  h2_ele_EtaMnEtaTrueVsPt = bookH2(iBooker, "EtaMnEtaTrueVsPt","ele momentum  eta - gen  eta vs pt",pt_nbin,0.,pt_max,deta_nbin/2,deta_min,deta_max);
  h1_ele_PhiMnPhiTrue = bookH1withSumw2(iBooker, "PhiMnPhiTrue","ele momentum  phi - gen  phi",dphi_nbin,dphi_min,dphi_max,"#phi_{rec} - #phi_{gen} (rad)");
  h1_ele_PhiMnPhiTrue_barrel = bookH1withSumw2(iBooker, "PhiMnPhiTrue_barrel","ele momentum  phi - gen  phi barrel",dphi_nbin,dphi_min,dphi_max,"#phi_{rec} - #phi_{gen} (rad)");
  h1_ele_PhiMnPhiTrue_endcaps = bookH1withSumw2(iBooker, "PhiMnPhiTrue_endcaps","ele momentum  phi - gen  phi endcaps",dphi_nbin,dphi_min,dphi_max,"#phi_{rec} - #phi_{gen} (rad)");
  h1_ele_PhiMnPhiTrue2 = bookH1(iBooker, "PhiMnPhiTrue2","ele momentum  phi - gen  phi",dphimatch2D_nbin,dphimatch_min,dphimatch_max);
  h2_ele_PhiMnPhiTrueVsEta = bookH2(iBooker, "PhiMnPhiTrueVsEta","ele momentum  phi - gen  phi vs eta",eta2D_nbin,eta_min,eta_max,dphi_nbin/2,dphi_min,dphi_max);
  h2_ele_PhiMnPhiTrueVsPhi = bookH2(iBooker, "PhiMnPhiTrueVsPhi","ele momentum  phi - gen  phi vs phi",phi2D_nbin,phi_min,phi_max,dphi_nbin/2,dphi_min,dphi_max);
  h2_ele_PhiMnPhiTrueVsPt = bookH2(iBooker, "PhiMnPhiTrueVsPt","ele momentum  phi - gen  phi vs pt",pt2D_nbin,0.,pt_max,dphi_nbin/2,dphi_min,dphi_max);
  h1_ele_ecalEnergyError = bookH1withSumw2(iBooker, "ecalEnergyError","Regression estimate of the ECAL energy error",error_nbin,0,enerror_max);
  h1_ele_ecalEnergyError_barrel = bookH1withSumw2(iBooker, "ecalEnergyError_barrel","Regression estimate of the ECAL energy error - barrel",30,0,30);
  h1_ele_ecalEnergyError_endcaps = bookH1withSumw2(iBooker, "ecalEnergyError_endcaps","Regression estimate of the ECAL energy error - endcaps",error_nbin,0,enerror_max);
  h1_ele_combinedP4Error = bookH1withSumw2(iBooker, "combinedP4Error","Estimated error on the combined momentum",error_nbin,0,enerror_max);
  h1_ele_combinedP4Error_barrel = bookH1withSumw2(iBooker, "combinedP4Error_barrel","Estimated error on the combined momentum - barrel",30,0,30);
  h1_ele_combinedP4Error_endcaps = bookH1withSumw2(iBooker, "combinedP4Error_endcaps","Estimated error on the combined momentum - endcaps",error_nbin,0,enerror_max);

  // matched electron, superclusters
  setBookPrefix("h_scl") ;
  h1_scl_En = bookH1withSumw2(iBooker, "energy","ele ecal energy",p_nbin,0.,p_max);
  h1_scl_EoEtrue_barrel = bookH1withSumw2(iBooker, "EoEtrue_barrel","ele ecal energy / gen energy, barrel",50,0.2,1.2,"E/E_{gen}");
  h1_scl_EoEtrue_barrel_etagap = bookH1withSumw2(iBooker, "EoEtrue_barrel_etagap","ele ecal energy / gen energy, barrel, etagap",50,0.2,1.2,"E/E_{gen}");
  h1_scl_EoEtrue_barrel_phigap = bookH1withSumw2(iBooker, "EoEtrue_barrel_phigap","ele ecal energy / gen energy, barrel, phigap",50,0.2,1.2,"E/E_{gen}");
  h1_scl_EoEtrue_ebeegap = bookH1withSumw2(iBooker, "EoEtrue_ebeegap","ele ecal energy / gen energy, ebeegap",50,0.2,1.2,"E/E_{gen}");
  h1_scl_EoEtrue_endcaps = bookH1withSumw2(iBooker, "EoEtrue_endcaps","ele ecal energy / gen energy, endcaps",50,0.2,1.2,"E/E_{gen}");
  h1_scl_EoEtrue_endcaps_deegap = bookH1withSumw2(iBooker, "EoEtrue_endcaps_deegap","ele ecal energy / gen energy, endcaps, deegap",50,0.2,1.2,"E/E_{gen}");
  h1_scl_EoEtrue_endcaps_ringgap = bookH1withSumw2(iBooker, "EoEtrue_endcaps_ringgap","ele ecal energy / gen energy, endcaps, ringgap",50,0.2,1.2,"E/E_{gen}");
  h1_scl_EoEtrue_barrel_new = bookH1withSumw2(iBooker, "EoEtrue_barrel_new","ele ecal energy / gen energy, barrel",poptrue_nbin,poptrue_min,poptrue_max,"E/E_{gen}");
  h1_scl_EoEtrue_barrel_new_etagap = bookH1withSumw2(iBooker, "EoEtrue_barrel_new_etagap","ele ecal energy / gen energy, barrel, etagap",poptrue_nbin,poptrue_min,poptrue_max,"E/E_{gen}");
  h1_scl_EoEtrue_barrel_new_phigap = bookH1withSumw2(iBooker, "EoEtrue_barrel_new_phigap","ele ecal energy / gen energy, barrel, phigap",poptrue_nbin,poptrue_min,poptrue_max,"E/E_{gen}");
  h1_scl_EoEtrue_ebeegap_new = bookH1withSumw2(iBooker, "EoEtrue_ebeegap_new","ele ecal energy / gen energy, ebeegap",poptrue_nbin,poptrue_min,poptrue_max,"E/E_{gen}");
  h1_scl_EoEtrue_endcaps_new = bookH1withSumw2(iBooker, "EoEtrue_endcaps_new","ele ecal energy / gen energy, endcaps",poptrue_nbin,poptrue_min,poptrue_max,"E/E_{gen}");
  h1_scl_EoEtrue_endcaps_new_deegap = bookH1withSumw2(iBooker, "EoEtrue_endcaps_new_deegap","ele ecal energy / gen energy, endcaps, deegap",poptrue_nbin,poptrue_min,poptrue_max,"E/E_{gen}");
  h1_scl_EoEtrue_endcaps_new_ringgap = bookH1withSumw2(iBooker, "EoEtrue_endcaps_new_ringgap","ele ecal energy / gen energy, endcaps, ringgap",poptrue_nbin,poptrue_min,poptrue_max,"E/E_{gen}");
  h1_scl_Et = bookH1withSumw2(iBooker, "et","ele supercluster transverse energy",pt_nbin,0.,pt_max);
  h2_scl_EtVsEta = bookH2(iBooker, "etVsEta","ele supercluster transverse energy vs eta",eta2D_nbin,eta_min,eta_max,pt_nbin,0.,pt_max);
  h2_scl_EtVsPhi = bookH2(iBooker, "etVsPhi","ele supercluster transverse energy vs phi",phi2D_nbin,phi_min,phi_max,pt_nbin,0.,pt_max);
  h2_scl_EtaVsPhi = bookH2(iBooker, "etaVsPhi","ele supercluster eta vs phi",phi2D_nbin,phi_min,phi_max,eta2D_nbin,eta_min,eta_max);
  h1_scl_Eta = bookH1withSumw2(iBooker, "eta","ele supercluster eta",eta_nbin,eta_min,eta_max);
  h1_scl_Phi = bookH1withSumw2(iBooker, "phi","ele supercluster phi",phi_nbin,phi_min,phi_max);
  h1_scl_SigEtaEta = bookH1withSumw2(iBooker, "sigetaeta","ele supercluster sigma eta eta",100,0.,0.05,"#sigma_{#eta #eta}","Events","ELE_LOGY E1 P");
  h1_scl_SigEtaEta_barrel = bookH1withSumw2(iBooker, "sigetaeta_barrel","ele supercluster sigma eta eta barrel",100,0.,0.05,"#sigma_{#eta #eta}","Events","ELE_LOGY E1 P");
  h1_scl_SigEtaEta_endcaps = bookH1withSumw2(iBooker, "sigetaeta_endcaps","ele supercluster sigma eta eta endcaps",100,0.,0.05,"#sigma_{#eta #eta}","Events","ELE_LOGY E1 P");
  h1_scl_SigIEtaIEta = bookH1withSumw2(iBooker, "sigietaieta","ele supercluster sigma ieta ieta",100,0.,0.05,"#sigma_{i#eta i#eta}","Events","ELE_LOGY E1 P");
  h1_scl_SigIEtaIEta_barrel = bookH1withSumw2(iBooker, "sigietaieta_barrel","ele supercluster sigma ieta ieta, barrel",100,0.,0.05,"#sigma_{i#eta i#eta}","Events","ELE_LOGY E1 P");
  h1_scl_SigIEtaIEta_endcaps = bookH1withSumw2(iBooker, "sigietaieta_endcaps","ele supercluster sigma ieta ieta, endcaps",100,0.,0.05,"#sigma_{i#eta i#eta}","Events","ELE_LOGY E1 P");
  h1_scl_full5x5_sigmaIetaIeta = bookH1withSumw2(iBooker, "full5x5_sigietaieta","ele supercluster full5x5 sigma ieta ieta",100,0.,0.05,"#sigma_{i#eta i#eta}","Events","ELE_LOGY E1 P");
  h1_scl_full5x5_sigmaIetaIeta_barrel = bookH1withSumw2(iBooker, "full5x5_sigietaieta_barrel","ele supercluster full5x5 sigma ieta ieta, barrel",100,0.,0.05,"#sigma_{i#eta i#eta}","Events","ELE_LOGY E1 P");
  h1_scl_full5x5_sigmaIetaIeta_endcaps = bookH1withSumw2(iBooker, "full5x5_sigietaieta_endcaps","ele supercluster full5x5 sigma ieta ieta, endcaps",100,0.,0.05,"#sigma_{i#eta i#eta}","Events","ELE_LOGY E1 P");
  h1_scl_E1x5 = bookH1withSumw2(iBooker, "E1x5","ele supercluster energy in 1x5",p_nbin,0., p_max,"E1x5 (GeV)","Events","ELE_LOGY E1 P");
  h1_scl_E1x5_barrel = bookH1withSumw2(iBooker, "E1x5_barrel","ele supercluster energy in 1x5 barrel",p_nbin,0., p_max,"E1x5 (GeV)","Events","ELE_LOGY E1 P");
  h1_scl_E1x5_endcaps = bookH1withSumw2(iBooker, "E1x5_endcaps","ele supercluster energy in 1x5 endcaps",p_nbin,0., p_max,"E1x5 (GeV)","Events","ELE_LOGY E1 P");
  h1_scl_E2x5max = bookH1withSumw2(iBooker, "E2x5max","ele supercluster energy in 2x5 max",p_nbin,0.,p_max,"E2x5 (GeV)","Events","ELE_LOGY E1 P");
  h1_scl_E2x5max_barrel = bookH1withSumw2(iBooker, "E2x5max_barrel","ele supercluster energy in 2x5 _max barrel",p_nbin,0.,p_max,"E2x5 (GeV)","Events","ELE_LOGY E1 P");
  h1_scl_E2x5max_endcaps = bookH1withSumw2(iBooker, "E2x5max_endcaps","ele supercluster energy in 2x5 _max endcaps",p_nbin,0.,p_max,"E2x5 (GeV)","Events","ELE_LOGY E1 P");
  h1_scl_E5x5 = bookH1withSumw2(iBooker, "E5x5","ele supercluster energy in 5x5",p_nbin,0.,p_max,"E5x5 (GeV)","Events","ELE_LOGY E1 P");
  h1_scl_E5x5_barrel = bookH1withSumw2(iBooker, "E5x5_barrel","ele supercluster energy in 5x5 barrel",p_nbin,0.,p_max,"E5x5 (GeV)","Events","ELE_LOGY E1 P");
  h1_scl_E5x5_endcaps = bookH1withSumw2(iBooker, "E5x5_endcaps","ele supercluster energy in 5x5 endcaps",p_nbin,0.,p_max,"E5x5 (GeV)","Events","ELE_LOGY E1 P");
  h2_scl_EoEtruePfVsEg = bookH2(iBooker, "EoEtruePfVsEg","ele supercluster energy / gen energy pflow vs eg",75,-0.1,1.4, 75, -0.1, 1.4,"E/E_{gen} (e/g)","E/E_{gen} (pflow)") ;
  h1_scl_bcl_EtotoEtrue = bookH1withSumw2(iBooker, "bcl_EtotoEtrue","Total basicclusters energy",50,0.2,1.2,"E/E_{gen}");
  h1_scl_bcl_EtotoEtrue_barrel = bookH1withSumw2(iBooker, "bcl_EtotoEtrue_barrel","Total basicclusters energy , barrel",50,0.2,1.2,"E/E_{gen}");
  h1_scl_bcl_EtotoEtrue_endcaps = bookH1withSumw2(iBooker, "bcl_EtotoEtrue_endcaps","Total basicclusters energy , endcaps",50,0.2,1.2,"E/E_{gen}");

  // matched electron, gsf tracks
  setBookPrefix("h_ele") ;
  h1_ele_ambiguousTracks = bookH1withSumw2(iBooker, "ambiguousTracks","ele # ambiguous tracks",  5,0.,5.,"N_{ambiguous tracks}","Events","ELE_LOGY E1 P");
  h2_ele_ambiguousTracksVsEta = bookH2(iBooker, "ambiguousTracksVsEta","ele # ambiguous tracks vs eta",eta2D_nbin,eta_min,eta_max,5,0.,5.);
  h2_ele_ambiguousTracksVsPhi = bookH2(iBooker, "ambiguousTracksVsPhi","ele # ambiguous tracks vs phi",phi2D_nbin,phi_min,phi_max,5,0.,5.);
  h2_ele_ambiguousTracksVsPt = bookH2(iBooker, "ambiguousTracksVsPt","ele # ambiguous tracks vs pt",pt2D_nbin,0.,pt_max,5,0.,5.);
  h1_ele_foundHits = bookH1withSumw2(iBooker, "foundHits","ele track # found hits",fhits_nbin,0.,fhits_max,"N_{hits}");
  h1_ele_foundHits_barrel = bookH1withSumw2(iBooker, "foundHits_barrel","ele track # found hits, barrel",fhits_nbin,0.,fhits_max,"N_{hits}");
  h1_ele_foundHits_endcaps = bookH1withSumw2(iBooker, "foundHits_endcaps","ele track # found hits, endcaps",fhits_nbin,0.,fhits_max,"N_{hits}");
  h2_ele_foundHitsVsEta = bookH2(iBooker, "foundHitsVsEta","ele track # found hits vs eta",eta2D_nbin,eta_min,eta_max,fhits_nbin,0.,fhits_max);
  h2_ele_foundHitsVsPhi = bookH2(iBooker, "foundHitsVsPhi","ele track # found hits vs phi",phi2D_nbin,phi_min,phi_max,fhits_nbin,0.,fhits_max);
  h2_ele_foundHitsVsPt = bookH2(iBooker, "foundHitsVsPt","ele track # found hits vs pt",pt2D_nbin,0.,pt_max,fhits_nbin,0.,fhits_max);
  h1_ele_lostHits = bookH1withSumw2(iBooker, "lostHits","ele track # lost hits",       5,0.,5.,"N_{lost hits}");
  h1_ele_lostHits_barrel = bookH1withSumw2(iBooker, "lostHits_barrel","ele track # lost hits, barrel",       5,0.,5.,"N_{lost hits}");
  h1_ele_lostHits_endcaps = bookH1withSumw2(iBooker, "lostHits_endcaps","ele track # lost hits, endcaps",       5,0.,5.,"N_{lost hits}");
  h2_ele_lostHitsVsEta = bookH2(iBooker, "lostHitsVsEta","ele track # lost hits vs eta",eta2D_nbin,eta_min,eta_max,lhits_nbin,0.,lhits_max);
  h2_ele_lostHitsVsPhi = bookH2(iBooker, "lostHitsVsPhi","ele track # lost hits vs eta",phi2D_nbin,phi_min,phi_max,lhits_nbin,0.,lhits_max);
  h2_ele_lostHitsVsPt = bookH2(iBooker, "lostHitsVsPt","ele track # lost hits vs eta",pt2D_nbin,0.,pt_max,lhits_nbin,0.,lhits_max);
  h1_ele_chi2 = bookH1withSumw2(iBooker, "chi2","ele track #chi^{2}",100,0.,15.,"#Chi^{2}","Events","ELE_LOGY E1 P");
  h1_ele_chi2_barrel = bookH1withSumw2(iBooker, "chi2_barrel","ele track #chi^{2}, barrel",100,0.,15.,"#Chi^{2}","Events","ELE_LOGY E1 P");
  h1_ele_chi2_endcaps = bookH1withSumw2(iBooker, "chi2_endcaps","ele track #chi^{2}, endcaps",100,0.,15.,"#Chi^{2}","Events","ELE_LOGY E1 P");
  h2_ele_chi2VsEta = bookH2(iBooker, "chi2VsEta","ele track #chi^{2} vs eta",eta2D_nbin,eta_min,eta_max,50,0.,15.);
  h2_ele_chi2VsPhi = bookH2(iBooker, "chi2VsPhi","ele track #chi^{2} vs phi",phi2D_nbin,phi_min,phi_max,50,0.,15.);
  h2_ele_chi2VsPt = bookH2(iBooker, "chi2VsPt","ele track #chi^{2} vs pt",pt2D_nbin,0.,pt_max,50,0.,15.);
  h1_ele_PinMnPout = bookH1withSumw2(iBooker, "PinMnPout","ele track inner p - outer p, mean of GSF components"   ,p_nbin,0.,200.,"P_{vertex} - P_{out} (GeV/c)");
  h1_ele_PinMnPout_mode = bookH1withSumw2(iBooker, "PinMnPout_mode","ele track inner p - outer p, mode of GSF components"   ,p_nbin,0.,100.,"P_{vertex} - P_{out}, mode of GSF components (GeV/c)");
  h2_ele_PinMnPoutVsEta_mode = bookH2(iBooker, "PinMnPoutVsEta_mode","ele track inner p - outer p vs eta, mode of GSF components" ,eta2D_nbin, eta_min,eta_max,p2D_nbin,0.,100.);
  h2_ele_PinMnPoutVsPhi_mode = bookH2(iBooker, "PinMnPoutVsPhi_mode","ele track inner p - outer p vs phi, mode of GSF components" ,phi2D_nbin, phi_min,phi_max,p2D_nbin,0.,100.);
  h2_ele_PinMnPoutVsPt_mode = bookH2(iBooker, "PinMnPoutVsPt_mode","ele track inner p - outer p vs pt, mode of GSF components" ,pt2D_nbin, 0.,pt_max,p2D_nbin,0.,100.);
  h2_ele_PinMnPoutVsE_mode = bookH2(iBooker, "PinMnPoutVsE_mode","ele track inner p - outer p vs E, mode of GSF components" ,p2D_nbin, 0.,200.,p2D_nbin,0.,100.);
  h2_ele_PinMnPoutVsChi2_mode = bookH2(iBooker, "PinMnPoutVsChi2_mode","ele track inner p - outer p vs track chi2, mode of GSF components" ,50, 0.,20.,p2D_nbin,0.,100.);
  h1_ele_outerP = bookH1withSumw2(iBooker, "outerP","ele track outer p, mean of GSF components",p_nbin,0.,p_max,"P_{out} (GeV/c)");
  h1_ele_outerP_mode = bookH1withSumw2(iBooker, "outerP_mode","ele track outer p, mode of GSF components",p_nbin,0.,p_max,"P_{out} (GeV/c)");
  h2_ele_outerPVsEta_mode = bookH2(iBooker, "outerPVsEta_mode","ele track outer p vs eta mode",eta2D_nbin,eta_min,eta_max,50,0.,p_max);
  h1_ele_outerPt = bookH1withSumw2(iBooker, "outerPt","ele track outer p_{T}, mean of GSF components",pt_nbin,0.,pt_max,"P_{T out} (GeV/c)");
  h1_ele_outerPt_mode = bookH1withSumw2(iBooker, "outerPt_mode","ele track outer p_{T}, mode of GSF components",pt_nbin,0.,pt_max,"P_{T out} (GeV/c)");
  h2_ele_outerPtVsEta_mode = bookH2(iBooker, "outerPtVsEta_mode","ele track outer p_{T} vs eta, mode of GSF components",eta2D_nbin,eta_min,eta_max,pt2D_nbin,0.,pt_max);
  h2_ele_outerPtVsPhi_mode = bookH2(iBooker, "outerPtVsPhi_mode","ele track outer p_{T} vs phi, mode of GSF components",phi2D_nbin,phi_min,phi_max,pt2D_nbin,0.,pt_max);
  h2_ele_outerPtVsPt_mode = bookH2(iBooker, "outerPtVsPt_mode","ele track outer p_{T} vs pt, mode of GSF components",pt2D_nbin,0.,100.,pt2D_nbin,0.,pt_max);

  // matched electrons, matching
  h1_ele_EoP = bookH1withSumw2(iBooker, "EoP","ele E/P_{vertex}",eop_nbin,0.,eop_max,"E/P_{vertex}","Events","ELE_LOGY E1 P");
  h1_ele_EoP_barrel = bookH1withSumw2(iBooker, "EoP_barrel","ele E/P_{vertex} barrel",eop_nbin,0.,eop_max,"E/P_{vertex}","Events","ELE_LOGY E1 P");
  h1_ele_EoP_endcaps = bookH1withSumw2(iBooker, "EoP_endcaps","ele E/P_{vertex} endcaps",eop_nbin,0.,eop_max,"E/P_{vertex}","Events","ELE_LOGY E1 P");
  h2_ele_EoPVsEta = bookH2(iBooker, "EoPVsEta","ele E/P_{vertex} vs eta",eta2D_nbin,eta_min,eta_max,eop2D_nbin,0.,eopmaxsht);
  h2_ele_EoPVsPhi = bookH2(iBooker, "EoPVsPhi","ele E/P_{vertex} vs phi",phi2D_nbin,phi_min,phi_max,eop2D_nbin,0.,eopmaxsht);
  h2_ele_EoPVsE = bookH2(iBooker, "EoPVsE","ele E/P_{vertex} vs E",  50,0.,p_max ,50,0.,5.);
  h1_ele_EseedOP = bookH1withSumw2(iBooker, "EseedOP","ele E_{seed}/P_{vertex}",eop_nbin,0.,eop_max,"E_{seed}/P_{vertex}","Events","ELE_LOGY E1 P");
  h1_ele_EseedOP_barrel = bookH1withSumw2(iBooker, "EseedOP_barrel","ele E_{seed}/P_{vertex} barrel",eop_nbin,0.,eop_max,"E_{seed}/P_{vertex}","Events","ELE_LOGY E1 P");
  h1_ele_EseedOP_endcaps = bookH1withSumw2(iBooker, "EseedOP_endcaps","ele E_{seed}/P_{vertex} endcaps",eop_nbin,0.,eop_max,"E_{seed}/P_{vertex}","Events","ELE_LOGY E1 P");
  h2_ele_EseedOPVsEta = bookH2(iBooker, "EseedOPVsEta","ele E_{seed}/P_{vertex} vs eta",eta2D_nbin,eta_min,eta_max,eop2D_nbin,0.,eopmaxsht);
  h2_ele_EseedOPVsPhi = bookH2(iBooker, "EseedOPVsPhi","ele E_{seed}/P_{vertex} vs phi",phi2D_nbin,phi_min,phi_max,eop2D_nbin,0.,eopmaxsht);
  h2_ele_EseedOPVsE = bookH2(iBooker, "EseedOPVsE","ele E_{seed}/P_{vertex} vs E",  50,0.,p_max ,50,0.,5.);
  h1_ele_EoPout = bookH1withSumw2(iBooker, "EoPout","ele E_{seed}/P_{out}",eop_nbin,0.,eop_max,"E_{seed}/P_{out}","Events","ELE_LOGY E1 P");
  h1_ele_EoPout_barrel = bookH1withSumw2(iBooker, "EoPout_barrel","ele E_{seed}/P_{out} barrel",eop_nbin,0.,eop_max,"E_{seed}/P_{out}","Events","ELE_LOGY E1 P");
  h1_ele_EoPout_endcaps = bookH1withSumw2(iBooker, "EoPout_endcaps","ele E_{seed}/P_{out} endcaps",eop_nbin,0.,eop_max,"E_{seed}/P_{out}","Events","ELE_LOGY E1 P");
  h2_ele_EoPoutVsEta = bookH2(iBooker, "EoPoutVsEta","ele E_{seed}/P_{out} vs eta",eta2D_nbin,eta_min,eta_max,eop2D_nbin,0.,eopmaxsht);
  h2_ele_EoPoutVsPhi = bookH2(iBooker, "EoPoutVsPhi","ele E_{seed}/P_{out} vs phi",phi2D_nbin,phi_min,phi_max,eop2D_nbin,0.,eopmaxsht);
  h2_ele_EoPoutVsE = bookH2(iBooker, "EoPoutVsE","ele E_{seed}/P_{out} vs E",p2D_nbin,0.,p_max,eop2D_nbin,0.,eopmaxsht);
  h1_ele_EeleOPout = bookH1withSumw2(iBooker, "EeleOPout","ele E_{ele}/P_{out}",eop_nbin,0.,eop_max,"E_{ele}/P_{out}","Events","ELE_LOGY E1 P");
  h1_ele_EeleOPout_barrel = bookH1withSumw2(iBooker, "EeleOPout_barrel","ele E_{ele}/P_{out} barrel",eop_nbin,0.,eop_max,"E_{ele}/P_{out}","Events","ELE_LOGY E1 P");
  h1_ele_EeleOPout_endcaps = bookH1withSumw2(iBooker, "EeleOPout_endcaps","ele E_{ele}/P_{out} endcaps",eop_nbin,0.,eop_max,"E_{ele}/P_{out}","Events","ELE_LOGY E1 P");
  h2_ele_EeleOPoutVsEta = bookH2(iBooker, "EeleOPoutVsEta","ele E_{ele}/P_{out} vs eta",eta2D_nbin,eta_min,eta_max,eop2D_nbin,0.,eopmaxsht);
  h2_ele_EeleOPoutVsPhi = bookH2(iBooker, "EeleOPoutVsPhi","ele E_{ele}/P_{out} vs phi",phi2D_nbin,phi_min,phi_max,eop2D_nbin,0.,eopmaxsht);
  h2_ele_EeleOPoutVsE = bookH2(iBooker, "EeleOPoutVsE","ele E_{ele}/P_{out} vs E",p2D_nbin,0.,p_max,eop2D_nbin,0.,eopmaxsht);
  h1_ele_dEtaSc_propVtx = bookH1withSumw2(iBooker, "dEtaSc_propVtx","ele #eta_{sc} - #eta_{tr}, prop from vertex",detamatch_nbin,detamatch_min,detamatch_max,"#eta_{sc} - #eta_{tr}","Events","ELE_LOGY E1 P");
  h1_ele_dEtaSc_propVtx_barrel = bookH1withSumw2(iBooker, "dEtaSc_propVtx_barrel","ele #eta_{sc} - #eta_{tr}, prop from vertex, barrel",detamatch_nbin,detamatch_min,detamatch_max,"#eta_{sc} - #eta_{tr}","Events","ELE_LOGY E1 P");
  h1_ele_dEtaSc_propVtx_endcaps = bookH1withSumw2(iBooker, "dEtaSc_propVtx_endcaps","ele #eta_{sc} - #eta_{tr}, prop from vertex, endcaps",detamatch_nbin,detamatch_min,detamatch_max,"#eta_{sc} - #eta_{tr}","Events","ELE_LOGY E1 P");
  h2_ele_dEtaScVsEta_propVtx = bookH2(iBooker, "dEtaScVsEta_propVtx","ele #eta_{sc} - #eta_{tr} vs eta, prop from vertex",eta2D_nbin,eta_min,eta_max,detamatch2D_nbin,detamatch_min,detamatch_max);
  h2_ele_dEtaScVsPhi_propVtx = bookH2(iBooker, "dEtaScVsPhi_propVtx","ele #eta_{sc} - #eta_{tr} vs phi, prop from vertex",phi2D_nbin,phi_min,phi_max,detamatch2D_nbin,detamatch_min,detamatch_max);
  h2_ele_dEtaScVsPt_propVtx = bookH2(iBooker, "dEtaScVsPt_propVtx","ele #eta_{sc} - #eta_{tr} vs pt, prop from vertex",pt2D_nbin,0.,pt_max,detamatch2D_nbin,detamatch_min,detamatch_max);
  h1_ele_dPhiSc_propVtx = bookH1withSumw2(iBooker, "dPhiSc_propVtx","ele #phi_{sc} - #phi_{tr}, prop from vertex",dphimatch_nbin,dphimatch_min,dphimatch_max,"#phi_{sc} - #phi_{tr} (rad)","Events","ELE_LOGY E1 P");
  h1_ele_dPhiSc_propVtx_barrel = bookH1withSumw2(iBooker, "dPhiSc_propVtx_barrel","ele #phi_{sc} - #phi_{tr}, prop from vertex, barrel",dphimatch_nbin,dphimatch_min,dphimatch_max,"#phi_{sc} - #phi_{tr} (rad)","Events","ELE_LOGY E1 P");
  h1_ele_dPhiSc_propVtx_endcaps = bookH1withSumw2(iBooker, "dPhiSc_propVtx_endcaps","ele #phi_{sc} - #phi_{tr}, prop from vertex, endcaps",dphimatch_nbin,dphimatch_min,dphimatch_max,"#phi_{sc} - #phi_{tr} (rad)","Events","ELE_LOGY E1 P");
  h2_ele_dPhiScVsEta_propVtx = bookH2(iBooker, "dPhiScVsEta_propVtx","ele #phi_{sc} - #phi_{tr} vs eta, prop from vertex",eta2D_nbin,eta_min,eta_max,dphimatch2D_nbin,dphimatch_min,dphimatch_max);
  h2_ele_dPhiScVsPhi_propVtx = bookH2(iBooker, "dPhiScVsPhi_propVtx","ele #phi_{sc} - #phi_{tr} vs phi, prop from vertex",phi2D_nbin,phi_min,phi_max,dphimatch2D_nbin,dphimatch_min,dphimatch_max);
  h2_ele_dPhiScVsPt_propVtx = bookH2(iBooker, "dPhiScVsPt_propVtx","ele #phi_{sc} - #phi_{tr} vs pt, prop from vertex",pt2D_nbin,0.,pt_max,dphimatch2D_nbin,dphimatch_min,dphimatch_max);
  h1_ele_dEtaCl_propOut = bookH1withSumw2(iBooker, "dEtaCl_propOut","ele #eta_{cl} - #eta_{tr}, prop from outermost",detamatch_nbin,detamatch_min,detamatch_max,"#eta_{seedcl} - #eta_{tr}","Events","ELE_LOGY E1 P");
  h1_ele_dEtaCl_propOut_barrel = bookH1withSumw2(iBooker, "dEtaCl_propOut_barrel","ele #eta_{cl} - #eta_{tr}, prop from outermost, barrel",detamatch_nbin,detamatch_min,detamatch_max,"#eta_{seedcl} - #eta_{tr}","Events","ELE_LOGY E1 P");
  h1_ele_dEtaCl_propOut_endcaps = bookH1withSumw2(iBooker, "dEtaCl_propOut_endcaps","ele #eta_{cl} - #eta_{tr}, prop from outermost, endcaps",detamatch_nbin,detamatch_min,detamatch_max,"#eta_{seedcl} - #eta_{tr}","Events","ELE_LOGY E1 P");
  h2_ele_dEtaClVsEta_propOut = bookH2(iBooker, "dEtaClVsEta_propOut","ele #eta_{cl} - #eta_{tr} vs eta, prop from out",eta2D_nbin,eta_min,eta_max,detamatch2D_nbin,detamatch_min,detamatch_max);
  h2_ele_dEtaClVsPhi_propOut = bookH2(iBooker, "dEtaClVsPhi_propOut","ele #eta_{cl} - #eta_{tr} vs phi, prop from out",phi2D_nbin,phi_min,phi_max,detamatch2D_nbin,detamatch_min,detamatch_max);
  h2_ele_dEtaClVsPt_propOut = bookH2(iBooker, "dEtaScVsPt_propOut","ele #eta_{cl} - #eta_{tr} vs pt, prop from out",pt2D_nbin,0.,pt_max,detamatch2D_nbin,detamatch_min,detamatch_max);
  h1_ele_dPhiCl_propOut = bookH1withSumw2(iBooker, "dPhiCl_propOut","ele #phi_{cl} - #phi_{tr}, prop from outermost",dphimatch_nbin,dphimatch_min,dphimatch_max,"#phi_{seedcl} - #phi_{tr} (rad)","Events","ELE_LOGY E1 P");
  h1_ele_dPhiCl_propOut_barrel = bookH1withSumw2(iBooker, "dPhiCl_propOut_barrel","ele #phi_{cl} - #phi_{tr}, prop from outermost, barrel",dphimatch_nbin,dphimatch_min,dphimatch_max,"#phi_{seedcl} - #phi_{tr} (rad)","Events","ELE_LOGY E1 P");
  h1_ele_dPhiCl_propOut_endcaps = bookH1withSumw2(iBooker, "dPhiCl_propOut_endcaps","ele #phi_{cl} - #phi_{tr}, prop from outermost, endcaps",dphimatch_nbin,dphimatch_min,dphimatch_max,"#phi_{seedcl} - #phi_{tr} (rad)","Events","ELE_LOGY E1 P");
  h2_ele_dPhiClVsEta_propOut = bookH2(iBooker, "dPhiClVsEta_propOut","ele #phi_{cl} - #phi_{tr} vs eta, prop from out",eta2D_nbin,eta_min,eta_max,dphimatch2D_nbin,dphimatch_min,dphimatch_max);
  h2_ele_dPhiClVsPhi_propOut = bookH2(iBooker, "dPhiClVsPhi_propOut","ele #phi_{cl} - #phi_{tr} vs phi, prop from out",phi2D_nbin,phi_min,phi_max,dphimatch2D_nbin,dphimatch_min,dphimatch_max);
  h2_ele_dPhiClVsPt_propOut = bookH2(iBooker, "dPhiSClsPt_propOut","ele #phi_{cl} - #phi_{tr} vs pt, prop from out",pt2D_nbin,0.,pt_max,dphimatch2D_nbin,dphimatch_min,dphimatch_max);
  h1_ele_dEtaEleCl_propOut = bookH1withSumw2(iBooker, "dEtaEleCl_propOut","ele #eta_{EleCl} - #eta_{tr}, prop from outermost",detamatch_nbin,detamatch_min,detamatch_max,"#eta_{elecl} - #eta_{tr}","Events","ELE_LOGY E1 P");
  h1_ele_dEtaEleCl_propOut_barrel = bookH1withSumw2(iBooker, "dEtaEleCl_propOut_barrel","ele #eta_{EleCl} - #eta_{tr}, prop from outermost, barrel",detamatch_nbin,detamatch_min,detamatch_max,"#eta_{elecl} - #eta_{tr}","Events","ELE_LOGY E1 P");
  h1_ele_dEtaEleCl_propOut_endcaps = bookH1withSumw2(iBooker, "dEtaEleCl_propOut_endcaps","ele #eta_{EleCl} - #eta_{tr}, prop from outermost, endcaps",detamatch_nbin,detamatch_min,detamatch_max,"#eta_{elecl} - #eta_{tr}","Events","ELE_LOGY E1 P");
  h2_ele_dEtaEleClVsEta_propOut = bookH2(iBooker, "dEtaEleClVsEta_propOut","ele #eta_{EleCl} - #eta_{tr} vs eta, prop from out",eta2D_nbin,eta_min,eta_max,detamatch2D_nbin,detamatch_min,detamatch_max);
  h2_ele_dEtaEleClVsPhi_propOut = bookH2(iBooker, "dEtaEleClVsPhi_propOut","ele #eta_{EleCl} - #eta_{tr} vs phi, prop from out",phi2D_nbin,phi_min,phi_max,detamatch2D_nbin,detamatch_min,detamatch_max);
  h2_ele_dEtaEleClVsPt_propOut = bookH2(iBooker, "dEtaScVsPt_propOut","ele #eta_{EleCl} - #eta_{tr} vs pt, prop from out",pt2D_nbin,0.,pt_max,detamatch2D_nbin,detamatch_min,detamatch_max);
  h1_ele_dPhiEleCl_propOut = bookH1withSumw2(iBooker, "dPhiEleCl_propOut","ele #phi_{EleCl} - #phi_{tr}, prop from outermost",dphimatch_nbin,dphimatch_min,dphimatch_max,"#phi_{elecl} - #phi_{tr} (rad)","Events","ELE_LOGY E1 P");
  h1_ele_dPhiEleCl_propOut_barrel = bookH1withSumw2(iBooker, "dPhiEleCl_propOut_barrel","ele #phi_{EleCl} - #phi_{tr}, prop from outermost, barrel",dphimatch_nbin,dphimatch_min,dphimatch_max,"#phi_{elecl} - #phi_{tr} (rad)","Events","ELE_LOGY E1 P");
  h1_ele_dPhiEleCl_propOut_endcaps = bookH1withSumw2(iBooker, "dPhiEleCl_propOut_endcaps","ele #phi_{EleCl} - #phi_{tr}, prop from outermost, endcaps",dphimatch_nbin,dphimatch_min,dphimatch_max,"#phi_{elecl} - #phi_{tr} (rad)","Events","ELE_LOGY E1 P");
  h2_ele_dPhiEleClVsEta_propOut = bookH2(iBooker, "dPhiEleClVsEta_propOut","ele #phi_{EleCl} - #phi_{tr} vs eta, prop from out",eta2D_nbin,eta_min,eta_max,dphimatch2D_nbin,dphimatch_min,dphimatch_max);
  h2_ele_dPhiEleClVsPhi_propOut = bookH2(iBooker, "dPhiEleClVsPhi_propOut","ele #phi_{EleCl} - #phi_{tr} vs phi, prop from out",phi2D_nbin,phi_min,phi_max,dphimatch2D_nbin,dphimatch_min,dphimatch_max);
  h2_ele_dPhiEleClVsPt_propOut = bookH2(iBooker, "dPhiSEleClsPt_propOut","ele #phi_{EleCl} - #phi_{tr} vs pt, prop from out",pt2D_nbin,0.,pt_max,dphimatch2D_nbin,dphimatch_min,dphimatch_max);
  h1_ele_HoE = bookH1withSumw2(iBooker, "HoE","ele hadronic energy / em energy",hoe_nbin, hoe_min, hoe_max,"H/E","Events","ELE_LOGY E1 P") ;
  h1_ele_HoE_barrel = bookH1withSumw2(iBooker, "HoE_barrel","ele hadronic energy / em energy, barrel",hoe_nbin, hoe_min, hoe_max,"H/E","Events","ELE_LOGY E1 P") ;
  h1_ele_HoE_endcaps = bookH1withSumw2(iBooker, "HoE_endcaps","ele hadronic energy / em energy, endcaps",hoe_nbin, hoe_min, hoe_max,"H/E","Events","ELE_LOGY E1 P") ;
  h1_ele_HoE_bc = bookH1withSumw2(iBooker, "HoE_bc","ele hadronic energy / em energy behind cluster",hoe_nbin, hoe_min, hoe_max,"H/E","Events","ELE_LOGY E1 P") ;
  h1_ele_HoE_bc_barrel = bookH1withSumw2(iBooker, "HoE_bc_barrel","ele hadronic energy / em energy, behind cluster barrel",hoe_nbin, hoe_min, hoe_max,"H/E","Events","ELE_LOGY E1 P") ;
  h1_ele_HoE_bc_endcaps = bookH1withSumw2(iBooker, "HoE_bc_endcaps","ele hadronic energy / em energy, behind cluster, endcaps",hoe_nbin, hoe_min, hoe_max,"H/E","Events","ELE_LOGY E1 P") ;
  h1_ele_hcalDepth1OverEcalBc = bookH1withSumw2(iBooker, "hcalDepth1OverEcalBc","hcalDepth1OverEcalBc",hoe_nbin, hoe_min, hoe_max,"H/E","Events","ELE_LOGY E1 P");
  h1_ele_hcalDepth1OverEcalBc_barrel = bookH1withSumw2(iBooker, "hcalDepth1OverEcalBc_barrel","hcalDepth1OverEcalBc_barrel",hoe_nbin, hoe_min, hoe_max,"H/E","Events","ELE_LOGY E1 P");
  h1_ele_hcalDepth1OverEcalBc_endcaps = bookH1withSumw2(iBooker, "hcalDepth1OverEcalBc_endcaps","hcalDepth1OverEcalBc_endcaps",hoe_nbin, hoe_min, hoe_max,"H/E","Events","ELE_LOGY E1 P");
  h1_ele_hcalDepth2OverEcalBc = bookH1withSumw2(iBooker, "hcalDepth2OverEcalBc","hcalDepth2OverEcalBc",hoe_nbin, hoe_min, hoe_max,"H/E","Events","ELE_LOGY E1 P");
  h1_ele_hcalDepth2OverEcalBc_barrel = bookH1withSumw2(iBooker, "hcalDepth2OverEcalBc_barrel","hcalDepth2OverEcalBc_barrel",hoe_nbin, hoe_min, hoe_max,"H/E","Events","ELE_LOGY E1 P");
  h1_ele_hcalDepth2OverEcalBc_endcaps = bookH1withSumw2(iBooker, "hcalDepth2OverEcalBc_endcaps","hcalDepth2OverEcalBc_endcaps",hoe_nbin, hoe_min, hoe_max,"H/E","Events","ELE_LOGY E1 P");

  h1_ele_HoE_fiducial = bookH1withSumw2(iBooker, "HoE_fiducial","ele hadronic energy / em energy, fiducial region",hoe_nbin, hoe_min, hoe_max,"H/E","Events","ELE_LOGY E1 P") ;
  h2_ele_HoEVsEta = bookH2(iBooker, "HoEVsEta","ele hadronic energy / em energy vs eta",eta_nbin,eta_min,eta_max,hoe_nbin, hoe_min, hoe_max) ;
  h2_ele_HoEVsPhi = bookH2(iBooker, "HoEVsPhi","ele hadronic energy / em energy vs phi",phi2D_nbin,phi_min,phi_max,hoe_nbin, hoe_min, hoe_max) ;
  h2_ele_HoEVsE = bookH2(iBooker, "HoEVsE","ele hadronic energy / em energy vs E",p_nbin, 0.,300.,hoe_nbin, hoe_min, hoe_max) ;

  // seeds
  h1_ele_seed_subdet2 = bookH1withSumw2(iBooker, "seedSubdet2","ele seed subdet 2nd layer",11,-0.5,10.5,"2nd hit subdet Id") ;
  h1_ele_seed_mask = bookH1withSumw2(iBooker, "seedMask","ele seed hits mask",13,-0.5,12.5) ;
  h1_ele_seed_mask_bpix = bookH1withSumw2(iBooker, "seedMask_Bpix","ele seed hits mask when subdet2 is bpix",13,-0.5,12.5) ;
  h1_ele_seed_mask_fpix = bookH1withSumw2(iBooker, "seedMask_Fpix","ele seed hits mask when subdet2 is fpix",13,-0.5,12.5) ;
  h1_ele_seed_mask_tec = bookH1withSumw2(iBooker, "seedMask_Tec","ele seed hits mask when subdet2 is tec",13,-0.5,12.5) ;
  h1_ele_seed_dphi2 = bookH1withSumw2(iBooker, "seedDphi2","ele seed dphi 2nd layer", 50,-0.010,+0.010,"#phi_{hit}-#phi_{pred} (rad)") ;
  h2_ele_seed_dphi2VsEta = bookH2(iBooker, "seedDphi2_VsEta","ele seed dphi 2nd layer vs eta",eta2D_nbin,eta_min,eta_max,50,-0.003,+0.003) ;
  h2_ele_seed_dphi2VsPt = bookH2(iBooker, "seedDphi2_VsPt","ele seed dphi 2nd layer vs pt",pt2D_nbin,0.,pt_max,50,-0.003,+0.003) ;
  h1_ele_seed_dphi2pos = bookH1withSumw2(iBooker, "seedDphi2Pos","ele seed dphi 2nd layer positron", 50,-0.010,+0.010,"#phi_{hit}-#phi_{pred} (rad)") ;
  h2_ele_seed_dphi2posVsEta = bookH2(iBooker, "seedDphi2Pos_VsEta","ele seed dphi 2nd layer positron vs eta",eta2D_nbin,eta_min,eta_max,50,-0.003,+0.003) ;
  h2_ele_seed_dphi2posVsPt = bookH2(iBooker, "seedDphi2Pos_VsPt","ele seed dphi 2nd layer positron vs pt",pt2D_nbin,0.,pt_max,50,-0.003,+0.003) ;
  h1_ele_seed_drz2 = bookH1withSumw2(iBooker, "seedDrz2","ele seed dr (dz) 2nd layer", 50,-0.03,+0.03,"r(z)_{hit}-r(z)_{pred} (cm)") ;
  h2_ele_seed_drz2VsEta = bookH2(iBooker, "seedDrz2_VsEta","ele seed dr/dz 2nd layer vs eta",eta2D_nbin,eta_min,eta_max,50,-0.03,+0.03) ;
  h2_ele_seed_drz2VsPt = bookH2(iBooker, "seedDrz2_VsPt","ele seed dr/dz 2nd layer vs pt",pt2D_nbin,0.,pt_max,50,-0.03,+0.03) ;
  h1_ele_seed_drz2pos = bookH1withSumw2(iBooker, "seedDrz2Pos","ele seed dr (dz) 2nd layer positron", 50,-0.03,+0.03,"r(z)_{hit}-r(z)_{pred} (cm)") ;
  h2_ele_seed_drz2posVsEta = bookH2(iBooker, "seedDrz2Pos_VsEta","ele seed dr/dz 2nd layer positron vs eta",eta2D_nbin,eta_min,eta_max,50,-0.03,+0.03) ;
  h2_ele_seed_drz2posVsPt = bookH2(iBooker, "seedDrz2Pos_VsPt","ele seed dr/dz 2nd layer positron vs pt",pt2D_nbin,0.,pt_max,50,-0.03,+0.03) ;

  // classes
  h1_ele_classes = bookH1withSumw2(iBooker, "classes","ele classes",20,0.0,20.,"class Id");
  h1_ele_eta = bookH1withSumw2(iBooker, "eta","ele electron eta",eta_nbin/2,0.0,eta_max);
  h1_ele_eta_golden = bookH1withSumw2(iBooker, "eta_golden","ele electron eta golden",eta_nbin/2,0.0,eta_max);
  h1_ele_eta_bbrem = bookH1withSumw2(iBooker, "eta_bbrem","ele electron eta bbrem",eta_nbin/2,0.0,eta_max);
  h1_ele_eta_shower = bookH1withSumw2(iBooker, "eta_shower","ele electron eta showering",eta_nbin/2,0.0,eta_max);
  h2_ele_PinVsPoutGolden_mode = bookH2(iBooker, "PinVsPoutGolden_mode","ele track inner p vs outer p vs eta, golden, mode of GSF components" ,p2D_nbin,0.,p_max,50,0.,p_max);
  h2_ele_PinVsPoutShowering_mode = bookH2(iBooker, "PinVsPoutShowering_mode","ele track inner p vs outer p vs eta, showering, mode of GSF components" ,p2D_nbin,0.,p_max,50,0.,p_max);
  h2_ele_PinVsPoutGolden_mean = bookH2(iBooker, "PinVsPoutGolden_mean","ele track inner p vs outer p vs eta, golden, mean of GSF components" ,p2D_nbin,0.,p_max,50,0.,p_max);
  h2_ele_PinVsPoutShowering_mean = bookH2(iBooker, "PinVsPoutShowering_mean","ele track inner p vs outer p vs eta, showering, mean of GSF components" ,p2D_nbin,0.,p_max,50,0.,p_max);
  h2_ele_PtinVsPtoutGolden_mode = bookH2(iBooker, "PtinVsPtoutGolden_mode","ele track inner pt vs outer pt vs eta, golden, mode of GSF components" ,pt2D_nbin,0.,pt_max,50,0.,pt_max);
  h2_ele_PtinVsPtoutShowering_mode = bookH2(iBooker, "PtinVsPtoutShowering_mode","ele track inner pt vs outer pt vs eta, showering, mode of GSF components" ,pt2D_nbin,0.,pt_max,50,0.,pt_max);
  h2_ele_PtinVsPtoutGolden_mean = bookH2(iBooker, "PtinVsPtoutGolden_mean","ele track inner pt vs outer pt vs eta, golden, mean of GSF components" ,pt2D_nbin,0.,pt_max,50,0.,pt_max);
  h2_ele_PtinVsPtoutShowering_mean = bookH2(iBooker, "PtinVsPtoutShowering_mean","ele track inner pt vs outer pt vs eta, showering, mean of GSF components" ,pt2D_nbin,0.,pt_max,50,0.,pt_max);
  setBookPrefix("h_scl") ;
  h1_scl_EoEtrueGolden_barrel = bookH1withSumw2(iBooker, "EoEtrue_golden_barrel","ele supercluster energy / gen energy, golden, barrel",poptrue_nbin,poptrue_min,poptrue_max);
  h1_scl_EoEtrueGolden_endcaps = bookH1withSumw2(iBooker, "EoEtrue_golden_endcaps","ele supercluster energy / gen energy, golden, endcaps",poptrue_nbin,poptrue_min,poptrue_max);
  h1_scl_EoEtrueShowering_barrel = bookH1withSumw2(iBooker, "EoEtrue_showering_barrel","ele supercluster energy / gen energy, showering, barrel",poptrue_nbin,poptrue_min,poptrue_max);
  h1_scl_EoEtrueShowering_endcaps = bookH1withSumw2(iBooker, "EoEtrue_showering_endcaps","ele supercluster energy / gen energy, showering, endcaps",poptrue_nbin,poptrue_min,poptrue_max);

  // isolation
  setBookPrefix("h_ele") ;
  h1_ele_tkSumPt_dr03 = bookH1withSumw2(iBooker, "tkSumPt_dr03","tk isolation sum, dR=0.3",100,0.0,20.,"TkIsoSum, cone 0.3 (GeV/c)","Events","ELE_LOGY E1 P");
  h1_ele_tkSumPt_dr03_barrel = bookH1withSumw2(iBooker, "tkSumPt_dr03_barrel","tk isolation sum, dR=0.3, barrel",100,0.0,20.,"TkIsoSum, cone 0.3 (GeV/c)","Events","ELE_LOGY E1 P");
  h1_ele_tkSumPt_dr03_endcaps = bookH1withSumw2(iBooker, "tkSumPt_dr03_endcaps","tk isolation sum, dR=0.3, endcaps",100,0.0,20.,"TkIsoSum, cone 0.3 (GeV/c)","Events","ELE_LOGY E1 P");
  h1_ele_ecalRecHitSumEt_dr03 = bookH1withSumw2(iBooker, "ecalRecHitSumEt_dr03","ecal isolation sum, dR=0.3",100,0.0,20.,"EcalIsoSum, cone 0.3 (GeV)","Events","ELE_LOGY E1 P");
  h1_ele_ecalRecHitSumEt_dr03_barrel = bookH1withSumw2(iBooker, "ecalRecHitSumEt_dr03_barrel","ecal isolation sum, dR=0.3, barrel",100,0.0,20.,"EcalIsoSum, cone 0.3 (GeV)","Events","ELE_LOGY E1 P");
  h1_ele_ecalRecHitSumEt_dr03_endcaps = bookH1withSumw2(iBooker, "ecalRecHitSumEt_dr03_endcaps","ecal isolation sum, dR=0.3, endcaps",100,0.0,20.,"EcalIsoSum, cone 0.3 (GeV)","Events","ELE_LOGY E1 P");
  h1_ele_hcalTowerSumEt_dr03_depth1 = bookH1withSumw2(iBooker, "hcalTowerSumEt_dr03_depth1","hcal depth1 isolation sum, dR=0.3",100,0.0,20.,"Hcal1IsoSum, cone 0.3 (GeV)","Events","ELE_LOGY E1 P");
  h1_ele_hcalTowerSumEt_dr03_depth1_barrel = bookH1withSumw2(iBooker, "hcalTowerSumEt_dr03_depth1_barrel","hcal depth1 isolation sum, dR=0.3, barrel",100,0.0,20.,"Hcal1IsoSum, cone 0.3 (GeV)","Events","ELE_LOGY E1 P");
  h1_ele_hcalTowerSumEt_dr03_depth1_endcaps = bookH1withSumw2(iBooker, "hcalTowerSumEt_dr03_depth1_endcaps","hcal depth1 isolation sum, dR=0.3, endcaps",100,0.0,20.,"Hcal1IsoSum, cone 0.3 (GeV)","Events","ELE_LOGY E1 P");
  h1_ele_hcalTowerSumEt_dr03_depth2 = bookH1withSumw2(iBooker, "hcalTowerSumEt_dr03_depth2","hcal depth2 isolation sum, dR=0.3",100,0.0,20.,"Hcal2IsoSum, cone 0.3 (GeV)","Events","ELE_LOGY E1 P");
  h1_ele_hcalTowerSumEt_dr03_depth2_barrel = bookH1withSumw2(iBooker, "hcalTowerSumEt_dr03_depth2_barrel","hcal depth2 isolation sum, dR=0.3",100,0.0,20.,"Hcal2IsoSum, cone 0.3 (GeV)","Events","ELE_LOGY E1 P");
  h1_ele_hcalTowerSumEt_dr03_depth2_endcaps = bookH1withSumw2(iBooker, "hcalTowerSumEt_dr03_depth2_endcaps","hcal depth2 isolation sum, dR=0.3",100,0.0,20.,"Hcal2IsoSum, cone 0.3 (GeV)","Events","ELE_LOGY E1 P");
  h1_ele_tkSumPt_dr04 = bookH1withSumw2(iBooker, "tkSumPt_dr04","tk isolation sum, dR=0.4",100,0.0,20.,"TkIsoSum, cone 0.4 (GeV/c)","Events","ELE_LOGY E1 P");
  h1_ele_tkSumPt_dr04_barrel = bookH1withSumw2(iBooker, "tkSumPt_dr04_barrel","tk isolation sum, dR=0.4, barrel",100,0.0,20.,"TkIsoSum, cone 0.4 (GeV/c)","Events","ELE_LOGY E1 P");
  h1_ele_tkSumPt_dr04_endcaps = bookH1withSumw2(iBooker, "tkSumPt_dr04_endcaps","tk isolation sum, dR=0.4, endcaps",100,0.0,20.,"TkIsoSum, cone 0.4 (GeV/c)","Events","ELE_LOGY E1 P");
  h1_ele_ecalRecHitSumEt_dr04 = bookH1withSumw2(iBooker, "ecalRecHitSumEt_dr04","ecal isolation sum, dR=0.4",100,0.0,20.,"EcalIsoSum, cone 0.4 (GeV)","Events","ELE_LOGY E1 P");
  h1_ele_ecalRecHitSumEt_dr04_barrel = bookH1withSumw2(iBooker, "ecalRecHitSumEt_dr04_barrel","ecal isolation sum, dR=0.4, barrel",100,0.0,20.,"EcalIsoSum, cone 0.4 (GeV)","Events","ELE_LOGY E1 P");
  h1_ele_ecalRecHitSumEt_dr04_endcaps = bookH1withSumw2(iBooker, "ecalRecHitSumEt_dr04_endcaps","ecal isolation sum, dR=0.4, endcaps",100,0.0,20.,"EcalIsoSum, cone 0.4 (GeV)","Events","ELE_LOGY E1 P");
  h1_ele_hcalTowerSumEt_dr04_depth1 = bookH1withSumw2(iBooker, "hcalTowerSumEt_dr04_depth1","hcal depth1 isolation sum, dR=0.4",100,0.0,20.,"Hcal1IsoSum, cone 0.4 (GeV)","Events","ELE_LOGY E1 P");
  h1_ele_hcalTowerSumEt_dr04_depth1_barrel = bookH1withSumw2(iBooker, "hcalTowerSumEt_dr04_depth1_barrel","hcal depth1 isolation sum, dR=0.4, barrel",100,0.0,20.,"Hcal1IsoSum, cone 0.4 (GeV)","Events","ELE_LOGY E1 P");
  h1_ele_hcalTowerSumEt_dr04_depth1_endcaps = bookH1withSumw2(iBooker, "hcalTowerSumEt_dr04_depth1_endcaps","hcal depth1 isolation sum, dR=0.4, endcaps",100,0.0,20.,"Hcal1IsoSum, cone 0.4 (GeV)","Events","ELE_LOGY E1 P");
  h1_ele_hcalTowerSumEt_dr04_depth2 = bookH1withSumw2(iBooker, "hcalTowerSumEt_dr04_depth2","hcal depth2 isolation sum, dR=0.4",100,0.0,20.,"Hcal2IsoSum, cone 0.4 (GeV)","Events","ELE_LOGY E1 P");
  h1_ele_hcalTowerSumEt_dr04_depth2_barrel = bookH1withSumw2(iBooker, "hcalTowerSumEt_dr04_depth2_barrel","hcal depth2 isolation sum, dR=0.4",100,0.0,20.,"Hcal2IsoSum, cone 0.4 (GeV)","Events","ELE_LOGY E1 P");
  h1_ele_hcalTowerSumEt_dr04_depth2_endcaps = bookH1withSumw2(iBooker, "hcalTowerSumEt_dr04_depth2_endcaps","hcal depth2 isolation sum, dR=0.4",100,0.0,20.,"Hcal2IsoSum, cone 0.4 (GeV)","Events","ELE_LOGY E1 P");

  // newHCAL
    // isolation new hcal
  h1_ele_hcalTowerSumEtBc_dr03_depth1 = bookH1withSumw2(iBooker, "hcalTowerSumEtBc_dr03_depth1","hcal depth1 isolation sum behind cluster, dR=0.3",100,0.0,20.,"Hcal1IsoSum, cone 0.3 (GeV)","Events","ELE_LOGY E1 P");
  h1_ele_hcalTowerSumEtBc_dr03_depth1_barrel = bookH1withSumw2(iBooker, "hcalTowerSumEtBc_dr03_depth1_barrel","hcal depth1 isolation sum behind cluster, dR=0.3, barrel",100,0.0,20.,"Hcal1IsoSum, cone 0.3 (GeV)","Events","ELE_LOGY E1 P");
  h1_ele_hcalTowerSumEtBc_dr03_depth1_endcaps = bookH1withSumw2(iBooker, "hcalTowerSumEtBc_dr03_depth1_endcaps","hcal depth1 isolation sum behind cluster, dR=0.3, endcaps",100,0.0,20.,"Hcal1IsoSum, cone 0.3 (GeV)","Events","ELE_LOGY E1 P");

  h1_ele_hcalTowerSumEtBc_dr04_depth1 = bookH1withSumw2(iBooker, "hcalTowerSumEtBc_dr04_depth1","hcal depth1 isolation sum behind cluster, dR=0.4",100,0.0,20.,"Hcal1IsoSum, cone 0.4 (GeV)","Events","ELE_LOGY E1 P");
  h1_ele_hcalTowerSumEtBc_dr04_depth1_barrel = bookH1withSumw2(iBooker, "hcalTowerSumEtBc_dr04_depth1_barrel","hcal depth1 isolation sum behind cluster, dR=0.4, barrel",100,0.0,20.,"Hcal1IsoSum, cone 0.4 (GeV)","Events","ELE_LOGY E1 P");
  h1_ele_hcalTowerSumEtBc_dr04_depth1_endcaps = bookH1withSumw2(iBooker, "hcalTowerSumEtBc_dr04_depth1_endcaps","hcal depth1 isolation sum behind cluster, dR=0.4, endcaps",100,0.0,20.,"Hcal1IsoSum, cone 0.4 (GeV)","Events","ELE_LOGY E1 P");

  h1_ele_hcalTowerSumEtBc_dr03_depth2 = bookH1withSumw2(iBooker, "hcalTowerSumEtBc_dr03_depth2","hcal depth2 isolation sum behind cluster, dR=0.3",100,0.0,20.,"Hcal1IsoSum, cone 0.3 (GeV)","Events","ELE_LOGY E1 P");
  h1_ele_hcalTowerSumEtBc_dr03_depth2_barrel = bookH1withSumw2(iBooker, "hcalTowerSumEtBc_dr03_depth2_barrel","hcal depth2 isolation sum behind cluster, dR=0.3, barrel",100,0.0,20.,"Hcal1IsoSum, cone 0.3 (GeV)","Events","ELE_LOGY E1 P");
  h1_ele_hcalTowerSumEtBc_dr03_depth2_endcaps = bookH1withSumw2(iBooker, "hcalTowerSumEtBc_dr03_depth2_endcaps","hcal depth2 isolation sum behind cluster, dR=0.3, endcaps",100,0.0,20.,"Hcal1IsoSum, cone 0.3 (GeV)","Events","ELE_LOGY E1 P");

  h1_ele_hcalTowerSumEtBc_dr04_depth2 = bookH1withSumw2(iBooker, "hcalTowerSumEtBc_dr04_depth2","hcal depth2 isolation sum behind cluster, dR=0.4",100,0.0,20.,"Hcal1IsoSum, cone 0.4 (GeV)","Events","ELE_LOGY E1 P");
  h1_ele_hcalTowerSumEtBc_dr04_depth2_barrel = bookH1withSumw2(iBooker, "hcalTowerSumEtBc_dr04_depth2_barrel","hcal depth2 isolation sum behind cluster, dR=0.4, barrel",100,0.0,20.,"Hcal1IsoSum, cone 0.4 (GeV)","Events","ELE_LOGY E1 P");
  h1_ele_hcalTowerSumEtBc_dr04_depth2_endcaps = bookH1withSumw2(iBooker, "hcalTowerSumEtBc_dr04_depth2_endcaps","hcal depth2 isolation sum behind cluster, dR=0.4, endcaps",100,0.0,20.,"Hcal1IsoSum, cone 0.4 (GeV)","Events","ELE_LOGY E1 P");

  // fbrem
  h1_ele_fbrem = bookH1withSumw2(iBooker, "fbrem","ele brem fraction, mode of GSF components",100,0.,1.,"P_{in} - P_{out} / P_{in}");
  h1_ele_fbrem_barrel = bookH1withSumw2(iBooker, "fbrem_barrel","ele brem fraction for barrel, mode of GSF components", 100, 0.,1.,"P_{in} - P_{out} / P_{in}");
  h1_ele_fbrem_endcaps = bookH1withSumw2(iBooker, "fbrem_endcaps", "ele brem franction for endcaps, mode of GSF components", 100, 0.,1.,"P_{in} - P_{out} / P_{in}");
  h1_ele_superclusterfbrem = bookH1withSumw2(iBooker, "superclusterfbrem","supercluster brem fraction, mode of GSF components",100,0.,1.,"P_{in} - P_{out} / P_{in}");
  h1_ele_superclusterfbrem_barrel = bookH1withSumw2(iBooker, "superclusterfbrem_barrel","supercluster brem fraction for barrel, mode of GSF components", 100, 0.,1.,"P_{in} - P_{out} / P_{in}");
  h1_ele_superclusterfbrem_endcaps = bookH1withSumw2(iBooker, "superclusterfbrem_endcaps", "supercluster brem franction for endcaps, mode of GSF components", 100, 0.,1.,"P_{in} - P_{out} / P_{in}");
  p1_ele_fbremVsEta_mode  = bookP1(iBooker, "fbremvsEtamode","mean ele brem fraction vs eta, mode of GSF components",eta2D_nbin,eta_min,eta_max,0.,1.,"#eta","<P_{in} - P_{out} / P_{in}>");
  p1_ele_fbremVsEta_mean  = bookP1(iBooker, "fbremvsEtamean","mean ele brem fraction vs eta, mean of GSF components",eta2D_nbin,eta_min,eta_max,0.,1.,"#eta","<P_{in} - P_{out} / P_{in}>");
  h1_ele_chargeInfo = bookH1withSumw2(iBooker, "chargeInfo","chargeInfo",5,-2.,3.);

  // e/g et pflow electrons
  h1_ele_mva = bookH1withSumw2(iBooker, "mva","ele identification mva",100,-1.,1.);
  h1_ele_mva_barrel = bookH1withSumw2(iBooker, "mva_barrel", "ele identification mva barrel",100,-1.,1.);
  h1_ele_mva_endcaps = bookH1withSumw2(iBooker, "mva_endcaps", "ele identification mva endcaps",100,-1.,1.);
  h1_ele_mva_isolated = bookH1withSumw2(iBooker, "mva_isolated","ele identification mva isolated",100,-1.,1.);
  h1_ele_mva_barrel_isolated = bookH1withSumw2(iBooker, "mva_isolated_barrel", "ele identification mva isolated barrel",100,-1.,1.);
  h1_ele_mva_endcaps_isolated = bookH1withSumw2(iBooker, "mva_isolated_endcaps", "ele identification mva isolated endcaps",100,-1.,1.);
  h1_ele_provenance = bookH1withSumw2(iBooker, "provenance","ele provenance",5,-2.,3.);
  h1_ele_provenance_barrel = bookH1withSumw2(iBooker, "provenance_barrel","ele provenance barrel",5,-2.,3.);
  h1_ele_provenance_endcaps = bookH1withSumw2(iBooker, "provenance_endcaps","ele provenance endcaps",5,-2.,3.);

  // pflow isolation variables
  h1_ele_chargedHadronIso = bookH1withSumw2(iBooker, "chargedHadronIso","chargedHadronIso",100,0.0,20.,"chargedHadronIso","Events","ELE_LOGY E1 P");
  h1_ele_chargedHadronIso_barrel = bookH1withSumw2(iBooker, "chargedHadronIso_barrel","chargedHadronIso for barrel",100,0.0,20.,"chargedHadronIso_barrel","Events","ELE_LOGY E1 P");
  h1_ele_chargedHadronIso_endcaps = bookH1withSumw2(iBooker, "chargedHadronIso_endcaps","chargedHadronIso for endcaps",100,0.0,20.,"chargedHadronIso_endcaps","Events","ELE_LOGY E1 P");
  h1_ele_neutralHadronIso = bookH1withSumw2(iBooker, "neutralHadronIso","neutralHadronIso",21,0.0,20.,"neutralHadronIso","Events", "ELE_LOGY E1 P");
  h1_ele_neutralHadronIso_barrel = bookH1withSumw2(iBooker, "neutralHadronIso_barrel","neutralHadronIso for barrel",21,0.0,20.,"neutralHadronIso_barrel","Events","ELE_LOGY E1 P");
  h1_ele_neutralHadronIso_endcaps = bookH1withSumw2(iBooker, "neutralHadronIso_endcaps","neutralHadronIso for endcaps",21,0.0,20.,"neutralHadronIso_endcaps","Events","ELE_LOGY E1 P");
  h1_ele_photonIso = bookH1withSumw2(iBooker, "photonIso","photonIso",100,0.0,20.,"photonIso","Events","ELE_LOGY E1 P");
  h1_ele_photonIso_barrel = bookH1withSumw2(iBooker, "photonIso_barrel","photonIso for barrel",100,0.0,20.,"photonIso_barrel","Events","ELE_LOGY E1 P");
  h1_ele_photonIso_endcaps = bookH1withSumw2(iBooker, "photonIso_endcaps","photonIso for endcaps",100,0.0,20.,"photonIso_endcaps","Events","ELE_LOGY E1 P");
  // -- pflow over pT
  h1_ele_chargedHadronRelativeIso = bookH1withSumw2(iBooker, "chargedHadronRelativeIso","chargedHadronRelativeIso",100,0.0,2.,"chargedHadronRelativeIso","Events","ELE_LOGY E1 P");
  h1_ele_chargedHadronRelativeIso_barrel = bookH1withSumw2(iBooker, "chargedHadronRelativeIso_barrel","chargedHadronRelativeIso for barrel",100,0.0,2.,"chargedHadronRelativeIso_barrel","Events","ELE_LOGY E1 P");
  h1_ele_chargedHadronRelativeIso_endcaps = bookH1withSumw2(iBooker, "chargedHadronRelativeIso_endcaps","chargedHadronRelativeIso for endcaps",100,0.0,2.,"chargedHadronRelativeIso_endcaps","Events","ELE_LOGY E1 P");
  h1_ele_neutralHadronRelativeIso = bookH1withSumw2(iBooker, "neutralHadronRelativeIso","neutralHadronRelativeIso",100,0.0,2.,"neutralHadronRelativeIso","Events","ELE_LOGY E1 P");
  h1_ele_neutralHadronRelativeIso_barrel = bookH1withSumw2(iBooker, "neutralHadronRelativeIso_barrel","neutralHadronRelativeIso for barrel",100,0.0,2.,"neutralHadronRelativeIso_barrel","Events","ELE_LOGY E1 P");
  h1_ele_neutralHadronRelativeIso_endcaps = bookH1withSumw2(iBooker, "neutralHadronRelativeIso_endcaps","neutralHadronRelativeIso for endcaps",100,0.0,2.,"neutralHadronRelativeIso_endcaps","Events","ELE_LOGY E1 P");
  h1_ele_photonRelativeIso = bookH1withSumw2(iBooker, "photonRelativeIso","photonRelativeIso",100,0.0,2.,"photonRelativeIso","Events","ELE_LOGY E1 P");
  h1_ele_photonRelativeIso_barrel = bookH1withSumw2(iBooker, "photonRelativeIso_barrel","photonRelativeIso for barrel",100,0.0,2.,"photonRelativeIso_barrel","Events","ELE_LOGY E1 P");
  h1_ele_photonRelativeIso_endcaps = bookH1withSumw2(iBooker, "photonRelativeIso_endcaps","photonRelativeIso for endcaps",100,0.0,2.,"photonRelativeIso_endcaps","Events","ELE_LOGY E1 P");

  // conversion rejection information
  h1_ele_convFlags = bookH1withSumw2(iBooker, "convFlags","conversion rejection flag",5,-1.5,3.5);
  h1_ele_convFlags_all = bookH1withSumw2(iBooker, "convFlags_all","conversion rejection flag, all electrons",5,-1.5,3.5);
  h1_ele_convDist = bookH1withSumw2(iBooker, "convDist","distance to the conversion partner",100,-15.,15.);
  h1_ele_convDist_all = bookH1withSumw2(iBooker, "convDist_all","distance to the conversion partner, all electrons",100,-15.,15.);
  h1_ele_convDcot = bookH1withSumw2(iBooker, "convDcot","difference of cot(angle) with the conversion partner",100,-CLHEP::pi/2.,CLHEP::pi/2.);
  h1_ele_convDcot_all = bookH1withSumw2(iBooker, "convDcot_all","difference of cot(angle) with the conversion partner, all electrons",100,-CLHEP::pi/2.,CLHEP::pi/2.);
  h1_ele_convRadius = bookH1withSumw2(iBooker, "convRadius","signed conversion radius",100,0.,130.);
  h1_ele_convRadius_all = bookH1withSumw2(iBooker, "convRadius_all","signed conversion radius, all electrons",100,0.,130.);

 }

ElectronMcSignalValidator::~ElectronMcSignalValidator()
 {}

void ElectronMcSignalValidator::analyze( const edm::Event & iEvent, const edm::EventSetup & iSetup )
 {
  // get collections
  edm::Handle<GsfElectronCollection> gsfElectrons ;
  iEvent.getByToken(electronCollection_, gsfElectrons) ;
  edm::Handle<GsfElectronCoreCollection> gsfElectronCores ;
  iEvent.getByToken(electronCoreCollection_,gsfElectronCores) ;
  edm::Handle<GsfTrackCollection> gsfElectronTracks ;
  iEvent.getByToken(electronTrackCollection_,gsfElectronTracks) ;
  edm::Handle<ElectronSeedCollection> gsfElectronSeeds ;
  iEvent.getByToken(electronSeedCollection_,gsfElectronSeeds) ;
  edm::Handle<GenParticleCollection> genParticles ;
  iEvent.getByToken(mcTruthCollection_, genParticles) ;
  edm::Handle<reco::BeamSpot> theBeamSpot ;
  iEvent.getByToken(beamSpotTag_,theBeamSpot) ;

  edm::Handle<edm::ValueMap<double> > isoFromDepsTk03Handle;
  iEvent.getByToken(isoFromDepsTk03Tag_, isoFromDepsTk03Handle);

  edm::Handle<edm::ValueMap<double> > isoFromDepsTk04Handle;
  iEvent.getByToken(isoFromDepsTk04Tag_, isoFromDepsTk04Handle);

  edm::Handle<edm::ValueMap<double> > isoFromDepsEcalFull03Handle;
  iEvent.getByToken( isoFromDepsEcalFull03Tag_, isoFromDepsEcalFull03Handle);

  edm::Handle<edm::ValueMap<double> > isoFromDepsEcalFull04Handle;
  iEvent.getByToken( isoFromDepsEcalFull04Tag_, isoFromDepsEcalFull04Handle);

  edm::Handle<edm::ValueMap<double> > isoFromDepsEcalReduced03Handle;
  iEvent.getByToken( isoFromDepsEcalReduced03Tag_, isoFromDepsEcalReduced03Handle);

  edm::Handle<edm::ValueMap<double> > isoFromDepsEcalReduced04Handle;
  iEvent.getByToken( isoFromDepsEcalReduced04Tag_, isoFromDepsEcalReduced04Handle);

  edm::Handle<edm::ValueMap<double> > isoFromDepsHcal03Handle;
  iEvent.getByToken( isoFromDepsHcal03Tag_, isoFromDepsHcal03Handle);

  edm::Handle<edm::ValueMap<double> > isoFromDepsHcal04Handle;
  iEvent.getByToken( isoFromDepsHcal04Tag_, isoFromDepsHcal04Handle);

  edm::Handle<reco::VertexCollection> vertexCollectionHandle;
  iEvent.getByToken(offlineVerticesCollection_, vertexCollectionHandle);
  if(!vertexCollectionHandle.isValid()) 
  {edm::LogInfo("ElectronMcSignalValidator::analyze") << "vertexCollectionHandle KO" ;}
  else 
  {
      edm::LogInfo("ElectronMcSignalValidator::analyze") << "vertexCollectionHandle OK" ;
  }
  
  edm::LogInfo("ElectronMcSignalValidator::analyze")
    <<"Treating event "<<iEvent.id()
    <<" with "<<gsfElectrons.product()->size()<<" electrons" ;
  h1_recEleNum->Fill((*gsfElectrons).size()) ;
  h1_recCoreNum->Fill((*gsfElectronCores).size());
  h1_recTrackNum->Fill((*gsfElectronTracks).size());
  h1_recSeedNum->Fill((*gsfElectronSeeds).size());
  h1_recOfflineVertices->Fill((*vertexCollectionHandle).size());


  //===============================================
  // all rec electrons
  //===============================================

  reco::GsfElectronCollection::const_iterator gsfIter ;
  for ( gsfIter=gsfElectrons->begin() ; gsfIter!=gsfElectrons->end() ; gsfIter++ )
   {
    // preselect electrons
    if (gsfIter->pt()>maxPt_ || std::abs(gsfIter->eta())>maxAbsEta_) continue ;

    //
    h1_ele_EoverP_all->Fill(gsfIter->eSuperClusterOverP()) ;
    h1_ele_EseedOP_all->Fill(gsfIter->eSeedClusterOverP()) ;
    h1_ele_EoPout_all->Fill(gsfIter->eSeedClusterOverPout()) ;
    h1_ele_EeleOPout_all->Fill( gsfIter->eEleClusterOverPout()) ;
    h1_ele_dEtaSc_propVtx_all->Fill(gsfIter->deltaEtaSuperClusterTrackAtVtx()) ;
    h1_ele_dPhiSc_propVtx_all->Fill(gsfIter->deltaPhiSuperClusterTrackAtVtx()) ;
    h1_ele_dEtaCl_propOut_all->Fill(gsfIter->deltaEtaSeedClusterTrackAtCalo()) ;
    h1_ele_dPhiCl_propOut_all->Fill(gsfIter->deltaPhiSeedClusterTrackAtCalo()) ;
    h1_ele_HoE_all->Fill(gsfIter->hcalOverEcal()) ;
    h1_ele_HoE_bc_all->Fill(gsfIter->hcalOverEcalBc()) ;
    h1_ele_TIP_all->Fill( EleRelPoint(gsfIter->vertex(),theBeamSpot->position()).perp() );
    h1_ele_vertexEta_all->Fill( gsfIter->eta() );
    h1_ele_vertexPt_all->Fill( gsfIter->pt() );
    h1_ele_Et_all->Fill( gsfIter->ecalEnergy()/cosh(gsfIter->superCluster()->eta()));
    float enrj1=gsfIter->ecalEnergy();

    // mee
    reco::GsfElectronCollection::const_iterator gsfIter2 ;
    for ( gsfIter2=gsfIter+1 ; gsfIter2!=gsfElectrons->end() ; gsfIter2++ )
     {
      math::XYZTLorentzVector p12 = (*gsfIter).p4()+(*gsfIter2).p4();
      float mee2 = p12.Dot(p12);
      float enrj2=gsfIter2->ecalEnergy() ;
      h1_ele_mee_all->Fill(sqrt(mee2));
      h2_ele_E2mnE1vsMee_all->Fill(sqrt(mee2),enrj2-enrj1);
      if (gsfIter->ecalDrivenSeed() && gsfIter2->ecalDrivenSeed())
       { h2_ele_E2mnE1vsMee_egeg_all->Fill(sqrt(mee2),enrj2-enrj1) ; }
      if (gsfIter->charge()*gsfIter2->charge()<0.)
       {
        h1_ele_mee_os->Fill(sqrt(mee2));
        if (gsfIter->isEB() && gsfIter2->isEB()) { h1_ele_mee_os_ebeb->Fill(sqrt(mee2)) ; }
	      if ((gsfIter->isEB() && gsfIter2->isEE()) || (gsfIter->isEE() && gsfIter2->isEB())) h1_ele_mee_os_ebee -> Fill(sqrt(mee2));
        if (gsfIter->isEE() && gsfIter2->isEE()) { h1_ele_mee_os_eeee->Fill(sqrt(mee2)) ; }
        if
         ( (gsfIter->classification()==GsfElectron::GOLDEN && gsfIter2->classification()==GsfElectron::GOLDEN) ||
           (gsfIter->classification()==GsfElectron::GOLDEN && gsfIter2->classification()==GsfElectron::BIGBREM) ||
           (gsfIter->classification()==GsfElectron::BIGBREM && gsfIter2->classification()==GsfElectron::GOLDEN) ||
           (gsfIter->classification()==GsfElectron::BIGBREM && gsfIter2->classification()==GsfElectron::BIGBREM) )
         { h1_ele_mee_os_gg->Fill(sqrt(mee2)) ; }
        else if
         ( (gsfIter->classification()==GsfElectron::SHOWERING && gsfIter2->classification()==GsfElectron::SHOWERING) ||
           (gsfIter->classification()==GsfElectron::SHOWERING && gsfIter2->isGap()) ||
           (gsfIter->isGap() && gsfIter2->classification()==GsfElectron::SHOWERING) ||
           (gsfIter->isGap() && gsfIter2->isGap()) )
         { h1_ele_mee_os_bb->Fill(sqrt(mee2)) ; }
        else
         { h1_ele_mee_os_gb->Fill(sqrt(mee2)) ; }
       }
     }

    // conversion rejection
    int flags = gsfIter->convFlags() ;
    if (flags==-9999) { flags=-1 ; }
    h1_ele_convFlags_all->Fill(flags);
    if (flags>=0.)
     {
      h1_ele_convDist_all->Fill( gsfIter->convDist() );
      h1_ele_convDcot_all->Fill( gsfIter->convDcot() );
      h1_ele_convRadius_all->Fill( gsfIter->convRadius() );
     }
   }

  //===============================================
  // charge mis-ID
  //===============================================

  int mcNum=0, gamNum=0, eleNum=0 ;
  bool matchingID, matchingMotherID ;

  reco::GenParticleCollection::const_iterator mcIter ;
  for
   ( mcIter=genParticles->begin() ; mcIter!=genParticles->end() ; mcIter++ )
   {
    // select requested matching gen particle
    matchingID=false;
    for ( unsigned int i=0 ; i<matchingIDs_.size() ; i++ )
     {
      if ( mcIter->pdgId() == matchingIDs_[i] )
       { matchingID=true ; }
     }
    if (matchingID)
     {
      // select requested mother matching gen particle
      // always include single particle with no mother
      const Candidate * mother = mcIter->mother() ;
      matchingMotherID = false ;
      for ( unsigned int i=0 ; i<matchingMotherIDs_.size() ; i++ )
       {
        if ((mother == 0) || ((mother != 0) &&  mother->pdgId() == matchingMotherIDs_[i]) )
         { matchingMotherID = true ; }
       }
      if (matchingMotherID)
       {
        if ( mcIter->pt()>maxPt_ || std::abs(mcIter->eta())>maxAbsEta_ )
         { continue ; }
        // suppress the endcaps
        //if (std::abs(mcIter->eta()) > 1.5) continue;
        // select central z
        //if ( std::abs(mcIter->production_vertex()->position().z())>50.) continue;

        // looking for the best matching gsf electron
        bool okGsfFound = false ;
        double gsfOkRatio = 999999. ;

        // find best matched electron
        reco::GsfElectron bestGsfElectron ;
        reco::GsfElectronCollection::const_iterator gsfIter ;
        for
         ( gsfIter=gsfElectrons->begin() ; gsfIter!=gsfElectrons->end() ; gsfIter++ )
         {
          double dphi = gsfIter->phi()-mcIter->phi() ;
          if (std::abs(dphi)>CLHEP::pi)
           { dphi = dphi < 0? (CLHEP::twopi) + dphi : dphi - CLHEP::twopi ; }
          double deltaR = sqrt(pow((gsfIter->eta()-mcIter->eta()),2) + pow(dphi,2)) ;
          if ( deltaR < deltaR_ )
           {
            double mc_charge = mcIter->pdgId() == 11 ? -1. : 1. ;
            h1_ele_ChargeMnChargeTrue->Fill( std::abs(gsfIter->charge()-mc_charge));
            // require here a charge mismatch
            if
             ( ( (mcIter->pdgId() == 11) && (gsfIter->charge() > 0.) ) ||
               ( (mcIter->pdgId() == -11) && (gsfIter->charge() < 0.) ) )
             {
              double tmpGsfRatio = gsfIter->p()/mcIter->p();
              if ( std::abs(tmpGsfRatio-1) < std::abs(gsfOkRatio-1) )
               {
                gsfOkRatio = tmpGsfRatio;
                bestGsfElectron=*gsfIter;
                okGsfFound = true;
               }
             }
           }
         } // loop over rec ele to look for the best one

        // analysis when the mc track is found
        if (okGsfFound)
         {
          // generated distributions for matched electrons
          h1_mc_Pt_matched_qmisid->Fill( mcIter->pt() ) ;
          h1_mc_Phi_matched_qmisid->Fill( mcIter->phi() ) ;
          h1_mc_AbsEta_matched_qmisid->Fill( std::abs(mcIter->eta()) ) ;
          h1_mc_Eta_matched_qmisid->Fill( mcIter->eta() ) ;
          h1_mc_Z_matched_qmisid->Fill( mcIter->vz() ) ;
         }
       }
     }
   }

  //===============================================
  // association mc-reco
  //===============================================

  for ( mcIter=genParticles->begin() ; mcIter!=genParticles->end() ; mcIter++ )
   {
    // number of mc particles
    mcNum++ ;

    // counts photons
    if (mcIter->pdgId() == 22 )
     { gamNum++ ; }

    // select requested matching gen particle
    matchingID = false ;
    for ( unsigned int i=0 ; i<matchingIDs_.size() ; i++ )
     {
      if ( mcIter->pdgId() == matchingIDs_[i] )
       { matchingID=true ; }
     }
    if (!matchingID) continue ;

    // select requested mother matching gen particle
    // always include single particle with no mother
    const Candidate * mother = mcIter->mother() ;
    matchingMotherID = false ;
    for ( unsigned int i=0 ; i<matchingMotherIDs_.size() ; i++ )
     {
      if ( (mother == 0) || ((mother != 0) &&  mother->pdgId() == matchingMotherIDs_[i]) )
       { matchingMotherID = true ; }
     }
    if (!matchingMotherID) continue ;

    // electron preselection
    if (mcIter->pt()> maxPt_ || std::abs(mcIter->eta())> maxAbsEta_)
     { continue ; }

    // suppress the endcaps
    //if (std::abs(mcIter->eta()) > 1.5) continue;
    // select central z
    //if ( std::abs(mcIter->production_vertex()->position().z())>50.) continue;

    eleNum++;
    h1_mc_Eta->Fill( mcIter->eta() );
    h1_mc_AbsEta->Fill( std::abs(mcIter->eta()) );
    h1_mc_P->Fill( mcIter->p() );
    h1_mc_Pt->Fill( mcIter->pt() );
    h1_mc_Phi->Fill( mcIter->phi() );
    h1_mc_Z->Fill( mcIter->vz() );
    h2_mc_PtEta->Fill( mcIter->eta(),mcIter->pt() );

    // find best matched electron
    bool okGsfFound = false ;
    double gsfOkRatio = 999999. ;
    reco::GsfElectron bestGsfElectron ;
    reco::GsfElectronRef bestGsfElectronRef ;
    reco::GsfElectronCollection::const_iterator gsfIter ;
    reco::GsfElectronCollection::size_type iElectron ;
    for ( gsfIter=gsfElectrons->begin(), iElectron=0 ; gsfIter!=gsfElectrons->end() ; gsfIter++, iElectron++ )
     {
      double dphi = gsfIter->phi()-mcIter->phi() ;
      if (std::abs(dphi)>CLHEP::pi)
       { dphi = dphi < 0? (CLHEP::twopi) + dphi : dphi - CLHEP::twopi ; }
      double deltaR = sqrt(pow((gsfIter->eta()-mcIter->eta()),2) + pow(dphi,2));
      if ( deltaR < deltaR_ )
       {
        if ( ( (mcIter->pdgId() == 11) && (gsfIter->charge() < 0.) ) ||
             ( (mcIter->pdgId() == -11) && (gsfIter->charge() > 0.) ) )
         {
          double tmpGsfRatio = gsfIter->p()/mcIter->p() ;
          if ( std::abs(tmpGsfRatio-1) < std::abs(gsfOkRatio-1) )
           {
            gsfOkRatio = tmpGsfRatio;
            bestGsfElectron=*gsfIter;
            bestGsfElectronRef=reco::GsfElectronRef(gsfElectrons,iElectron);
            okGsfFound = true;
           }
         }
       }
     } // loop over rec ele to look for the best one
    if (! okGsfFound) continue ;

    //------------------------------------
    // analysis when the mc track is found
    //------------------------------------

    // electron related distributions
    h1_ele_charge->Fill( bestGsfElectron.charge() );
    h2_ele_chargeVsEta->Fill( bestGsfElectron.eta(),bestGsfElectron.charge() );
    h2_ele_chargeVsPhi->Fill( bestGsfElectron.phi(),bestGsfElectron.charge() );
    h2_ele_chargeVsPt->Fill( bestGsfElectron.pt(),bestGsfElectron.charge() );
    h1_ele_vertexP->Fill( bestGsfElectron.p() );
    h1_ele_vertexPt->Fill( bestGsfElectron.pt() );
    h1_ele_Et->Fill( bestGsfElectron.ecalEnergy()/cosh(bestGsfElectron.superCluster()->eta()));
    h2_ele_vertexPtVsEta->Fill(  bestGsfElectron.eta(),bestGsfElectron.pt() );
    h2_ele_vertexPtVsPhi->Fill(  bestGsfElectron.phi(),bestGsfElectron.pt() );
    h1_ele_vertexEta->Fill( bestGsfElectron.eta() );
    
    h2_scl_EoEtrueVsrecOfflineVertices->Fill( (*vertexCollectionHandle).size(), bestGsfElectron.ecalEnergy()/mcIter->p() );
    if (bestGsfElectron.isEB())  h2_scl_EoEtrueVsrecOfflineVertices_barrel->Fill( (*vertexCollectionHandle).size(),bestGsfElectron.ecalEnergy()/mcIter->p() );
    if (bestGsfElectron.isEE())  h2_scl_EoEtrueVsrecOfflineVertices_endcaps->Fill( (*vertexCollectionHandle).size(),bestGsfElectron.ecalEnergy()/mcIter->p() );
    
    // generated distributions for matched electrons
    h1_mc_Pt_matched->Fill( mcIter->pt() );
    h1_mc_Phi_matched->Fill( mcIter->phi() );
    h1_mc_AbsEta_matched->Fill( std::abs(mcIter->eta()) );
    h1_mc_Eta_matched->Fill( mcIter->eta() );
    h2_mc_PtEta_matched->Fill(  mcIter->eta(),mcIter->pt() );
    h2_ele_vertexEtaVsPhi->Fill(  bestGsfElectron.phi(),bestGsfElectron.eta() );
    h1_ele_vertexPhi->Fill( bestGsfElectron.phi() );
    h1_ele_vertexX->Fill( bestGsfElectron.vertex().x() );
    h1_ele_vertexY->Fill( bestGsfElectron.vertex().y() );
    h1_ele_vertexZ->Fill( bestGsfElectron.vertex().z() );
    h1_mc_Z_matched->Fill( mcIter->vz() );
    double d =
     (bestGsfElectron.vertex().x()-mcIter->vx())*(bestGsfElectron.vertex().x()-mcIter->vx()) +
     (bestGsfElectron.vertex().y()-mcIter->vy())*(bestGsfElectron.vertex().y()-mcIter->vy()) ;
    d = sqrt(d) ;
    h1_ele_vertexTIP->Fill( d );
    h2_ele_vertexTIPVsEta->Fill(  bestGsfElectron.eta(), d );
    h2_ele_vertexTIPVsPhi->Fill(  bestGsfElectron.phi(), d );
    h2_ele_vertexTIPVsPt->Fill(  bestGsfElectron.pt(), d );
    h1_ele_EtaMnEtaTrue->Fill( bestGsfElectron.eta()-mcIter->eta());
    if (bestGsfElectron.isEB()) h1_ele_EtaMnEtaTrue_barrel->Fill( bestGsfElectron.eta()-mcIter->eta());
    if (bestGsfElectron.isEE()) h1_ele_EtaMnEtaTrue_endcaps->Fill( bestGsfElectron.eta()-mcIter->eta());
    h2_ele_EtaMnEtaTrueVsEta->Fill( bestGsfElectron.eta(), bestGsfElectron.eta()-mcIter->eta());
    h2_ele_EtaMnEtaTrueVsPhi->Fill( bestGsfElectron.phi(), bestGsfElectron.eta()-mcIter->eta());
    h2_ele_EtaMnEtaTrueVsPt->Fill( bestGsfElectron.pt(), bestGsfElectron.eta()-mcIter->eta());
    h1_ele_PhiMnPhiTrue->Fill( bestGsfElectron.phi()-mcIter->phi());
    if (bestGsfElectron.isEB()) h1_ele_PhiMnPhiTrue_barrel->Fill( bestGsfElectron.phi()-mcIter->phi());
    if (bestGsfElectron.isEE()) h1_ele_PhiMnPhiTrue_endcaps->Fill( bestGsfElectron.phi()-mcIter->phi());
    h1_ele_PhiMnPhiTrue2->Fill( bestGsfElectron.phi()-mcIter->phi());
    h2_ele_PhiMnPhiTrueVsEta->Fill( bestGsfElectron.eta(), bestGsfElectron.phi()-mcIter->phi());
    h2_ele_PhiMnPhiTrueVsPhi->Fill( bestGsfElectron.phi(), bestGsfElectron.phi()-mcIter->phi());
    h2_ele_PhiMnPhiTrueVsPt->Fill( bestGsfElectron.pt(), bestGsfElectron.phi()-mcIter->phi());
    h1_ele_PoPtrue->Fill( bestGsfElectron.p()/mcIter->p());
    h1_ele_PtoPttrue->Fill( bestGsfElectron.pt()/mcIter->pt());
    h2_ele_PoPtrueVsEta->Fill( bestGsfElectron.eta(), bestGsfElectron.p()/mcIter->p());
    h2_ele_PoPtrueVsPhi->Fill( bestGsfElectron.phi(), bestGsfElectron.p()/mcIter->p());
    h2_ele_PoPtrueVsPt->Fill( bestGsfElectron.py(), bestGsfElectron.p()/mcIter->p());
    if (bestGsfElectron.isEB()) h1_ele_PoPtrue_barrel->Fill( bestGsfElectron.p()/mcIter->p());
    if (bestGsfElectron.isEE()) h1_ele_PoPtrue_endcaps->Fill( bestGsfElectron.p()/mcIter->p());
    if (bestGsfElectron.isEB() && bestGsfElectron.classification() == GsfElectron::GOLDEN) h1_ele_PoPtrue_golden_barrel->Fill( bestGsfElectron.p()/mcIter->p());
    if (bestGsfElectron.isEE() && bestGsfElectron.classification() == GsfElectron::GOLDEN) h1_ele_PoPtrue_golden_endcaps->Fill( bestGsfElectron.p()/mcIter->p());
    if (bestGsfElectron.isEB() && bestGsfElectron.classification() == GsfElectron::SHOWERING) h1_ele_PoPtrue_showering_barrel->Fill( bestGsfElectron.p()/mcIter->p());
    if (bestGsfElectron.isEE() && bestGsfElectron.classification() == GsfElectron::SHOWERING) h1_ele_PoPtrue_showering_endcaps->Fill( bestGsfElectron.p()/mcIter->p());
    if (bestGsfElectron.isEB()) h1_ele_PtoPttrue_barrel->Fill( bestGsfElectron.pt()/mcIter->pt());
    if (bestGsfElectron.isEE()) h1_ele_PtoPttrue_endcaps->Fill( bestGsfElectron.pt()/mcIter->pt());
    h1_ele_ecalEnergyError->Fill(bestGsfElectron.correctedEcalEnergyError());
    if (bestGsfElectron.isEB()) h1_ele_ecalEnergyError_barrel->Fill(bestGsfElectron.correctedEcalEnergyError());
    if (bestGsfElectron.isEE()) h1_ele_ecalEnergyError_endcaps->Fill(bestGsfElectron.correctedEcalEnergyError());
    h1_ele_combinedP4Error->Fill(bestGsfElectron.p4Error(bestGsfElectron.P4_COMBINATION));
    if (bestGsfElectron.isEB()) h1_ele_combinedP4Error_barrel->Fill(bestGsfElectron.p4Error(bestGsfElectron.P4_COMBINATION));
    if (bestGsfElectron.isEE()) h1_ele_combinedP4Error_endcaps->Fill(bestGsfElectron.p4Error(bestGsfElectron.P4_COMBINATION));

    // supercluster related distributions
    reco::SuperClusterRef sclRef = bestGsfElectron.superCluster();

    h1_scl_En->Fill(bestGsfElectron.ecalEnergy());
    if (bestGsfElectron.isEB())  h1_scl_EoEtrue_barrel->Fill(bestGsfElectron.ecalEnergy()/mcIter->p());
    if (bestGsfElectron.isEE())  h1_scl_EoEtrue_endcaps->Fill(bestGsfElectron.ecalEnergy()/mcIter->p());
    if (bestGsfElectron.isEB() && bestGsfElectron.isEBEtaGap())  h1_scl_EoEtrue_barrel_etagap->Fill(bestGsfElectron.ecalEnergy()/mcIter->p());
    if (bestGsfElectron.isEB() && bestGsfElectron.isEBPhiGap())  h1_scl_EoEtrue_barrel_phigap->Fill(bestGsfElectron.ecalEnergy()/mcIter->p());
    if (bestGsfElectron.isEBEEGap())  h1_scl_EoEtrue_ebeegap->Fill(bestGsfElectron.ecalEnergy()/mcIter->p());
    if (bestGsfElectron.isEE() && bestGsfElectron.isEEDeeGap())  h1_scl_EoEtrue_endcaps_deegap->Fill(bestGsfElectron.ecalEnergy()/mcIter->p());
    if (bestGsfElectron.isEE() && bestGsfElectron.isEERingGap())  h1_scl_EoEtrue_endcaps_ringgap->Fill(bestGsfElectron.ecalEnergy()/mcIter->p());
    if (bestGsfElectron.isEB())  h1_scl_EoEtrue_barrel_new->Fill(bestGsfElectron.ecalEnergy()/mcIter->p());
    if (bestGsfElectron.isEE())  h1_scl_EoEtrue_endcaps_new->Fill(bestGsfElectron.ecalEnergy()/mcIter->p());
    if (bestGsfElectron.isEB() && bestGsfElectron.isEBEtaGap())  h1_scl_EoEtrue_barrel_new_etagap->Fill(bestGsfElectron.ecalEnergy()/mcIter->p());
    if (bestGsfElectron.isEB() && bestGsfElectron.isEBPhiGap())  h1_scl_EoEtrue_barrel_new_phigap->Fill(bestGsfElectron.ecalEnergy()/mcIter->p());
    if (bestGsfElectron.isEBEEGap())  h1_scl_EoEtrue_ebeegap_new->Fill(bestGsfElectron.ecalEnergy()/mcIter->p());
    if (bestGsfElectron.isEE() && bestGsfElectron.isEEDeeGap())  h1_scl_EoEtrue_endcaps_new_deegap->Fill(bestGsfElectron.ecalEnergy()/mcIter->p());
    if (bestGsfElectron.isEE() && bestGsfElectron.isEERingGap())  h1_scl_EoEtrue_endcaps_new_ringgap->Fill(bestGsfElectron.ecalEnergy()/mcIter->p());

    double R=TMath::Sqrt(sclRef->x()*sclRef->x() + sclRef->y()*sclRef->y() +sclRef->z()*sclRef->z());
    double Rt=TMath::Sqrt(sclRef->x()*sclRef->x() + sclRef->y()*sclRef->y());
    h1_scl_Et->Fill(sclRef->energy()*(Rt/R));
    h2_scl_EtVsEta->Fill(sclRef->eta(),sclRef->energy()*(Rt/R));
    h2_scl_EtVsPhi->Fill(sclRef->phi(),sclRef->energy()*(Rt/R));
    h1_scl_Eta->Fill(sclRef->eta());
    h2_scl_EtaVsPhi->Fill(sclRef->phi(),sclRef->eta());
    h1_scl_Phi->Fill(sclRef->phi());
    h1_scl_SigEtaEta->Fill(bestGsfElectron.scSigmaEtaEta());
    if (bestGsfElectron.isEB()) h1_scl_SigEtaEta_barrel->Fill(bestGsfElectron.scSigmaEtaEta());
    if (bestGsfElectron.isEE()) h1_scl_SigEtaEta_endcaps->Fill(bestGsfElectron.scSigmaEtaEta());
    h1_scl_SigIEtaIEta->Fill(bestGsfElectron.scSigmaIEtaIEta());
    if (bestGsfElectron.isEB()) h1_scl_SigIEtaIEta_barrel->Fill(bestGsfElectron.scSigmaIEtaIEta());
    if (bestGsfElectron.isEE()) h1_scl_SigIEtaIEta_endcaps->Fill(bestGsfElectron.scSigmaIEtaIEta());
    h1_scl_full5x5_sigmaIetaIeta->Fill(bestGsfElectron.full5x5_sigmaIetaIeta());
    if (bestGsfElectron.isEB()) h1_scl_full5x5_sigmaIetaIeta_barrel->Fill(bestGsfElectron.full5x5_sigmaIetaIeta());
    if (bestGsfElectron.isEE()) h1_scl_full5x5_sigmaIetaIeta_endcaps->Fill(bestGsfElectron.full5x5_sigmaIetaIeta());
    h1_scl_E1x5->Fill(bestGsfElectron.scE1x5());
    if (bestGsfElectron.isEB()) h1_scl_E1x5_barrel->Fill(bestGsfElectron.scE1x5());
    if (bestGsfElectron.isEE()) h1_scl_E1x5_endcaps->Fill(bestGsfElectron.scE1x5());
    h1_scl_E2x5max->Fill(bestGsfElectron.scE2x5Max());
    if (bestGsfElectron.isEB()) h1_scl_E2x5max_barrel->Fill(bestGsfElectron.scE2x5Max());
    if (bestGsfElectron.isEE()) h1_scl_E2x5max_endcaps->Fill(bestGsfElectron.scE2x5Max());
    h1_scl_E5x5->Fill(bestGsfElectron.scE5x5());
    if (bestGsfElectron.isEB()) h1_scl_E5x5_barrel->Fill(bestGsfElectron.scE5x5());
    if (bestGsfElectron.isEE()) h1_scl_E5x5_endcaps->Fill(bestGsfElectron.scE5x5());
    float pfEnergy=0. ;
    if (!bestGsfElectron.parentSuperCluster().isNull()) pfEnergy = bestGsfElectron.parentSuperCluster()->energy();
    h2_scl_EoEtruePfVsEg->Fill(bestGsfElectron.ecalEnergy()/mcIter->p(),pfEnergy/mcIter->p());

    float Etot = 0.; 
    CaloCluster_iterator it = bestGsfElectron.superCluster()->clustersBegin();
    CaloCluster_iterator itend = bestGsfElectron.superCluster()->clustersEnd();
    for(; it !=itend;++it) {
        Etot += (*it)->energy();
    }
    h1_scl_bcl_EtotoEtrue->Fill( Etot/mcIter->p() );
    if (bestGsfElectron.isEB()) h1_scl_bcl_EtotoEtrue_barrel->Fill( Etot/mcIter->p() );
    if (bestGsfElectron.isEE()) h1_scl_bcl_EtotoEtrue_endcaps->Fill( Etot/mcIter->p() );
    
    // track related distributions
    h1_ele_ambiguousTracks->Fill( bestGsfElectron.ambiguousGsfTracksSize() );
    h2_ele_ambiguousTracksVsEta->Fill( bestGsfElectron.eta(), bestGsfElectron.ambiguousGsfTracksSize() );
    h2_ele_ambiguousTracksVsPhi->Fill( bestGsfElectron.phi(), bestGsfElectron.ambiguousGsfTracksSize() );
    h2_ele_ambiguousTracksVsPt->Fill( bestGsfElectron.pt(), bestGsfElectron.ambiguousGsfTracksSize() );
    if (!readAOD_) // track extra does not exist in AOD
     {
      h1_ele_foundHits->Fill( bestGsfElectron.gsfTrack()->numberOfValidHits() );
      if (bestGsfElectron.isEB()) h1_ele_foundHits_barrel->Fill( bestGsfElectron.gsfTrack()->numberOfValidHits() );
      if (bestGsfElectron.isEE()) h1_ele_foundHits_endcaps->Fill( bestGsfElectron.gsfTrack()->numberOfValidHits() );
      h2_ele_foundHitsVsEta->Fill( bestGsfElectron.eta(), bestGsfElectron.gsfTrack()->numberOfValidHits() );
      h2_ele_foundHitsVsPhi->Fill( bestGsfElectron.phi(), bestGsfElectron.gsfTrack()->numberOfValidHits() );
      h2_ele_foundHitsVsPt->Fill( bestGsfElectron.pt(), bestGsfElectron.gsfTrack()->numberOfValidHits() );
      h1_ele_lostHits->Fill( bestGsfElectron.gsfTrack()->numberOfLostHits() );
      if (bestGsfElectron.isEB()) h1_ele_lostHits_barrel->Fill( bestGsfElectron.gsfTrack()->numberOfLostHits() );
      if (bestGsfElectron.isEE()) h1_ele_lostHits_endcaps->Fill( bestGsfElectron.gsfTrack()->numberOfLostHits() );
      h2_ele_lostHitsVsEta->Fill( bestGsfElectron.eta(), bestGsfElectron.gsfTrack()->numberOfLostHits() );
      h2_ele_lostHitsVsPhi->Fill( bestGsfElectron.phi(), bestGsfElectron.gsfTrack()->numberOfLostHits() );
      h2_ele_lostHitsVsPt->Fill( bestGsfElectron.pt(), bestGsfElectron.gsfTrack()->numberOfLostHits() );
      h1_ele_chi2->Fill( bestGsfElectron.gsfTrack()->normalizedChi2() );
      if (bestGsfElectron.isEB()) h1_ele_chi2_barrel->Fill( bestGsfElectron.gsfTrack()->normalizedChi2() );
      if (bestGsfElectron.isEE()) h1_ele_chi2_endcaps->Fill( bestGsfElectron.gsfTrack()->normalizedChi2() );
      h2_ele_chi2VsEta->Fill( bestGsfElectron.eta(), bestGsfElectron.gsfTrack()->normalizedChi2() );
      h2_ele_chi2VsPhi->Fill( bestGsfElectron.phi(), bestGsfElectron.gsfTrack()->normalizedChi2() );
      h2_ele_chi2VsPt->Fill( bestGsfElectron.pt(), bestGsfElectron.gsfTrack()->normalizedChi2() );
     }
    // from gsf track interface, hence using mean
    if (!readAOD_) // track extra does not exist in AOD
     {
      h1_ele_PinMnPout->Fill( bestGsfElectron.gsfTrack()->innerMomentum().R() - bestGsfElectron.gsfTrack()->outerMomentum().R() );
      h1_ele_outerP->Fill( bestGsfElectron.gsfTrack()->outerMomentum().R() );
      h1_ele_outerPt->Fill( bestGsfElectron.gsfTrack()->outerMomentum().Rho() );
     }
    // from electron interface, hence using mode
    h1_ele_PinMnPout_mode->Fill( bestGsfElectron.trackMomentumAtVtx().R() - bestGsfElectron.trackMomentumOut().R() );
    h2_ele_PinMnPoutVsEta_mode->Fill(  bestGsfElectron.eta(), bestGsfElectron.trackMomentumAtVtx().R() - bestGsfElectron.trackMomentumOut().R() );
    h2_ele_PinMnPoutVsPhi_mode->Fill(  bestGsfElectron.phi(), bestGsfElectron.trackMomentumAtVtx().R() - bestGsfElectron.trackMomentumOut().R() );
    h2_ele_PinMnPoutVsPt_mode->Fill(  bestGsfElectron.pt(), bestGsfElectron.trackMomentumAtVtx().R() - bestGsfElectron.trackMomentumOut().R() );
    h2_ele_PinMnPoutVsE_mode->Fill(  bestGsfElectron.caloEnergy(), bestGsfElectron.trackMomentumAtVtx().R() - bestGsfElectron.trackMomentumOut().R() );
    if (!readAOD_)  // track extra does not exist in AOD
     {
      h2_ele_PinMnPoutVsChi2_mode->Fill
       ( bestGsfElectron.gsfTrack()->normalizedChi2(),
         bestGsfElectron.trackMomentumAtVtx().R() - bestGsfElectron.trackMomentumOut().R() ) ;
     }
    h1_ele_outerP_mode->Fill( bestGsfElectron.trackMomentumOut().R() );
    h2_ele_outerPVsEta_mode->Fill(bestGsfElectron.eta(),  bestGsfElectron.trackMomentumOut().R() );
    h1_ele_outerPt_mode->Fill( bestGsfElectron.trackMomentumOut().Rho() );
    h2_ele_outerPtVsEta_mode->Fill(bestGsfElectron.eta(),  bestGsfElectron.trackMomentumOut().Rho() );
    h2_ele_outerPtVsPhi_mode->Fill(bestGsfElectron.phi(),  bestGsfElectron.trackMomentumOut().Rho() );
    h2_ele_outerPtVsPt_mode->Fill(bestGsfElectron.pt(),  bestGsfElectron.trackMomentumOut().Rho() );

    if (!readAOD_) // track extra does not exist in AOD
     {
      edm::RefToBase<TrajectorySeed> seed = bestGsfElectron.gsfTrack()->extra()->seedRef();
      ElectronSeedRef elseed=seed.castTo<ElectronSeedRef>();
      h1_ele_seed_subdet2->Fill(elseed->subDet2());
      h1_ele_seed_mask->Fill(elseed->hitsMask());
      if (elseed->subDet2()==1)
       { h1_ele_seed_mask_bpix->Fill(elseed->hitsMask()); }
      else if (elseed->subDet2()==2)
       { h1_ele_seed_mask_fpix->Fill(elseed->hitsMask()); }
      else if (elseed->subDet2()==6)
       { h1_ele_seed_mask_tec->Fill(elseed->hitsMask()); }

      if ( elseed->dPhi2() != std::numeric_limits<float>::infinity() ) {
        h1_ele_seed_dphi2->Fill(elseed->dPhi2());
        h2_ele_seed_dphi2VsEta->Fill(bestGsfElectron.eta(), elseed->dPhi2());
        h2_ele_seed_dphi2VsPt->Fill(bestGsfElectron.pt(), elseed->dPhi2());
      }
      else {
      }
      if ( elseed->dPhi2Pos() != std::numeric_limits<float>::infinity() ) {
        h1_ele_seed_dphi2pos->Fill(elseed->dPhi2Pos());
        h2_ele_seed_dphi2posVsEta->Fill(bestGsfElectron.eta(), elseed->dPhi2Pos());
        h2_ele_seed_dphi2posVsPt->Fill(bestGsfElectron.pt(), elseed->dPhi2Pos());
      }
      if ( elseed->dRz2() != std::numeric_limits<float>::infinity() ) {
        h1_ele_seed_drz2->Fill(elseed->dRz2());
        h2_ele_seed_drz2VsEta->Fill(bestGsfElectron.eta(), elseed->dRz2());
        h2_ele_seed_drz2VsPt->Fill(bestGsfElectron.pt(), elseed->dRz2());
      }
      if ( elseed->dRz2Pos() != std::numeric_limits<float>::infinity() ) {
        h1_ele_seed_drz2pos->Fill(elseed->dRz2Pos());
        h2_ele_seed_drz2posVsEta->Fill(bestGsfElectron.eta(), elseed->dRz2Pos());
        h2_ele_seed_drz2posVsPt->Fill(bestGsfElectron.pt(), elseed->dRz2Pos());
      }
     }

    // match distributions
    h1_ele_EoP->Fill( bestGsfElectron.eSuperClusterOverP() );
    if (bestGsfElectron.isEB()) h1_ele_EoP_barrel->Fill( bestGsfElectron.eSuperClusterOverP() );
    if (bestGsfElectron.isEE()) h1_ele_EoP_endcaps->Fill( bestGsfElectron.eSuperClusterOverP() );
    h2_ele_EoPVsEta->Fill(bestGsfElectron.eta(),  bestGsfElectron.eSuperClusterOverP() );
    h2_ele_EoPVsPhi->Fill(bestGsfElectron.phi(),  bestGsfElectron.eSuperClusterOverP() );
    h2_ele_EoPVsE->Fill(bestGsfElectron.caloEnergy(),  bestGsfElectron.eSuperClusterOverP() );
    h1_ele_EseedOP->Fill( bestGsfElectron.eSeedClusterOverP() );
    if (bestGsfElectron.isEB()) h1_ele_EseedOP_barrel->Fill( bestGsfElectron.eSeedClusterOverP() );
    if (bestGsfElectron.isEE()) h1_ele_EseedOP_endcaps->Fill( bestGsfElectron.eSeedClusterOverP() );
    h2_ele_EseedOPVsEta->Fill(bestGsfElectron.eta(),  bestGsfElectron.eSeedClusterOverP() );
    h2_ele_EseedOPVsPhi->Fill(bestGsfElectron.phi(),  bestGsfElectron.eSeedClusterOverP() );
    h2_ele_EseedOPVsE->Fill(bestGsfElectron.caloEnergy(),  bestGsfElectron.eSeedClusterOverP() );
    h1_ele_EoPout->Fill( bestGsfElectron.eSeedClusterOverPout() );
    if (bestGsfElectron.isEB()) h1_ele_EoPout_barrel->Fill( bestGsfElectron.eSeedClusterOverPout() );
    if (bestGsfElectron.isEE()) h1_ele_EoPout_endcaps->Fill( bestGsfElectron.eSeedClusterOverPout() );
    h2_ele_EoPoutVsEta->Fill( bestGsfElectron.eta(), bestGsfElectron.eSeedClusterOverPout() );
    h2_ele_EoPoutVsPhi->Fill( bestGsfElectron.phi(), bestGsfElectron.eSeedClusterOverPout() );
    h2_ele_EoPoutVsE->Fill( bestGsfElectron.caloEnergy(), bestGsfElectron.eSeedClusterOverPout() );
    h1_ele_EeleOPout->Fill( bestGsfElectron.eEleClusterOverPout() );
    if (bestGsfElectron.isEB()) h1_ele_EeleOPout_barrel->Fill( bestGsfElectron.eEleClusterOverPout() );
    if (bestGsfElectron.isEE()) h1_ele_EeleOPout_endcaps->Fill( bestGsfElectron.eEleClusterOverPout() );
    h2_ele_EeleOPoutVsEta->Fill( bestGsfElectron.eta(), bestGsfElectron.eEleClusterOverPout() );
    h2_ele_EeleOPoutVsPhi->Fill( bestGsfElectron.phi(), bestGsfElectron.eEleClusterOverPout() );
    h2_ele_EeleOPoutVsE->Fill( bestGsfElectron.caloEnergy(), bestGsfElectron.eEleClusterOverPout() );
    h1_ele_dEtaSc_propVtx->Fill(bestGsfElectron.deltaEtaSuperClusterTrackAtVtx());
    if (bestGsfElectron.isEB()) h1_ele_dEtaSc_propVtx_barrel->Fill(bestGsfElectron.deltaEtaSuperClusterTrackAtVtx());
    if (bestGsfElectron.isEE())h1_ele_dEtaSc_propVtx_endcaps->Fill(bestGsfElectron.deltaEtaSuperClusterTrackAtVtx());
    h2_ele_dEtaScVsEta_propVtx->Fill( bestGsfElectron.eta(),bestGsfElectron.deltaEtaSuperClusterTrackAtVtx());
    h2_ele_dEtaScVsPhi_propVtx->Fill(bestGsfElectron.phi(),bestGsfElectron.deltaEtaSuperClusterTrackAtVtx());
    h2_ele_dEtaScVsPt_propVtx->Fill(bestGsfElectron.pt(),bestGsfElectron.deltaEtaSuperClusterTrackAtVtx());
    h1_ele_dPhiSc_propVtx->Fill(bestGsfElectron.deltaPhiSuperClusterTrackAtVtx());
    if (bestGsfElectron.isEB()) h1_ele_dPhiSc_propVtx_barrel->Fill(bestGsfElectron.deltaPhiSuperClusterTrackAtVtx());
    if (bestGsfElectron.isEE())h1_ele_dPhiSc_propVtx_endcaps->Fill(bestGsfElectron.deltaPhiSuperClusterTrackAtVtx());
    h2_ele_dPhiScVsEta_propVtx->Fill( bestGsfElectron.eta(),bestGsfElectron.deltaPhiSuperClusterTrackAtVtx());
    h2_ele_dPhiScVsPhi_propVtx->Fill(bestGsfElectron.phi(),bestGsfElectron.deltaPhiSuperClusterTrackAtVtx());
    h2_ele_dPhiScVsPt_propVtx->Fill(bestGsfElectron.pt(),bestGsfElectron.deltaPhiSuperClusterTrackAtVtx());
    h1_ele_dEtaCl_propOut->Fill(bestGsfElectron.deltaEtaSeedClusterTrackAtCalo());
    if (bestGsfElectron.isEB()) h1_ele_dEtaCl_propOut_barrel->Fill(bestGsfElectron.deltaEtaSeedClusterTrackAtCalo());
    if (bestGsfElectron.isEE()) h1_ele_dEtaCl_propOut_endcaps->Fill(bestGsfElectron.deltaEtaSeedClusterTrackAtCalo());
    h2_ele_dEtaClVsEta_propOut->Fill( bestGsfElectron.eta(),bestGsfElectron.deltaEtaSeedClusterTrackAtCalo());
    h2_ele_dEtaClVsPhi_propOut->Fill(bestGsfElectron.phi(),bestGsfElectron.deltaEtaSeedClusterTrackAtCalo());
    h2_ele_dEtaClVsPt_propOut->Fill(bestGsfElectron.pt(),bestGsfElectron.deltaEtaSeedClusterTrackAtCalo());
    h1_ele_dPhiCl_propOut->Fill(bestGsfElectron.deltaPhiSeedClusterTrackAtCalo());
    if (bestGsfElectron.isEB()) h1_ele_dPhiCl_propOut_barrel->Fill(bestGsfElectron.deltaPhiSeedClusterTrackAtCalo());
    if (bestGsfElectron.isEE()) h1_ele_dPhiCl_propOut_endcaps->Fill(bestGsfElectron.deltaPhiSeedClusterTrackAtCalo());
    h2_ele_dPhiClVsEta_propOut->Fill( bestGsfElectron.eta(),bestGsfElectron.deltaPhiSeedClusterTrackAtCalo());
    h2_ele_dPhiClVsPhi_propOut->Fill(bestGsfElectron.phi(),bestGsfElectron.deltaPhiSeedClusterTrackAtCalo());
    h2_ele_dPhiClVsPt_propOut->Fill(bestGsfElectron.pt(),bestGsfElectron.deltaPhiSeedClusterTrackAtCalo());
    h1_ele_dEtaEleCl_propOut->Fill(bestGsfElectron.deltaEtaEleClusterTrackAtCalo());
    if (bestGsfElectron.isEB()) h1_ele_dEtaEleCl_propOut_barrel->Fill(bestGsfElectron.deltaEtaEleClusterTrackAtCalo());
    if (bestGsfElectron.isEE()) h1_ele_dEtaEleCl_propOut_endcaps->Fill(bestGsfElectron.deltaEtaEleClusterTrackAtCalo());
    h2_ele_dEtaEleClVsEta_propOut->Fill( bestGsfElectron.eta(),bestGsfElectron.deltaEtaEleClusterTrackAtCalo());
    h2_ele_dEtaEleClVsPhi_propOut->Fill(bestGsfElectron.phi(),bestGsfElectron.deltaEtaEleClusterTrackAtCalo());
    h2_ele_dEtaEleClVsPt_propOut->Fill(bestGsfElectron.pt(),bestGsfElectron.deltaEtaEleClusterTrackAtCalo());
    h1_ele_dPhiEleCl_propOut->Fill(bestGsfElectron.deltaPhiEleClusterTrackAtCalo());
    if (bestGsfElectron.isEB()) h1_ele_dPhiEleCl_propOut_barrel->Fill(bestGsfElectron.deltaPhiEleClusterTrackAtCalo());
    if (bestGsfElectron.isEE()) h1_ele_dPhiEleCl_propOut_endcaps->Fill(bestGsfElectron.deltaPhiEleClusterTrackAtCalo());
    h2_ele_dPhiEleClVsEta_propOut->Fill( bestGsfElectron.eta(),bestGsfElectron.deltaPhiEleClusterTrackAtCalo());
    h2_ele_dPhiEleClVsPhi_propOut->Fill(bestGsfElectron.phi(),bestGsfElectron.deltaPhiEleClusterTrackAtCalo());
    h2_ele_dPhiEleClVsPt_propOut->Fill(bestGsfElectron.pt(),bestGsfElectron.deltaPhiEleClusterTrackAtCalo());
    h1_ele_HoE->Fill(bestGsfElectron.hcalOverEcal());
    h1_ele_HoE_bc->Fill(bestGsfElectron.hcalOverEcalBc());
    if (bestGsfElectron.isEB()) h1_ele_HoE_bc_barrel->Fill(bestGsfElectron.hcalOverEcalBc());
    if (bestGsfElectron.isEE()) h1_ele_HoE_bc_endcaps->Fill(bestGsfElectron.hcalOverEcalBc());
    if (bestGsfElectron.isEB()) h1_ele_HoE_barrel->Fill(bestGsfElectron.hcalOverEcal());
    if (bestGsfElectron.isEE()) h1_ele_HoE_endcaps->Fill(bestGsfElectron.hcalOverEcal());
    if (!bestGsfElectron.isEBEtaGap() && !bestGsfElectron.isEBPhiGap()&& !bestGsfElectron.isEBEEGap() &&
        !bestGsfElectron.isEERingGap() && !bestGsfElectron.isEEDeeGap()) h1_ele_HoE_fiducial->Fill(bestGsfElectron.hcalOverEcal());
    h2_ele_HoEVsEta->Fill( bestGsfElectron.eta(),bestGsfElectron.hcalOverEcal());
    h2_ele_HoEVsPhi->Fill(bestGsfElectron.phi(),bestGsfElectron.hcalOverEcal());
    h2_ele_HoEVsE->Fill(bestGsfElectron.caloEnergy(),bestGsfElectron.hcalOverEcal());

    //classes
    int eleClass = bestGsfElectron.classification();
    if (bestGsfElectron.isEE()) eleClass+=10;
    h1_ele_classes->Fill(eleClass);

    if (bestGsfElectron.classification() == GsfElectron::GOLDEN && bestGsfElectron.isEB())  h1_scl_EoEtrueGolden_barrel->Fill(sclRef->energy()/mcIter->p());
    if (bestGsfElectron.classification() == GsfElectron::GOLDEN && bestGsfElectron.isEE())  h1_scl_EoEtrueGolden_endcaps->Fill(sclRef->energy()/mcIter->p());
    if (bestGsfElectron.classification() == GsfElectron::SHOWERING && bestGsfElectron.isEB())  h1_scl_EoEtrueShowering_barrel->Fill(sclRef->energy()/mcIter->p());
    if (bestGsfElectron.classification() == GsfElectron::SHOWERING && bestGsfElectron.isEE())  h1_scl_EoEtrueShowering_endcaps->Fill(sclRef->energy()/mcIter->p());

    //eleClass = eleClass%100; // get rid of barrel/endcap distinction
    h1_ele_eta->Fill(std::abs(bestGsfElectron.eta()));
    if (bestGsfElectron.classification() == GsfElectron::GOLDEN) h1_ele_eta_golden->Fill(std::abs(bestGsfElectron.eta()));
    if (bestGsfElectron.classification() == GsfElectron::BIGBREM) h1_ele_eta_bbrem->Fill(std::abs(bestGsfElectron.eta()));
    if (bestGsfElectron.classification() == GsfElectron::SHOWERING) h1_ele_eta_shower->Fill(std::abs(bestGsfElectron.eta()));

    // fbrem

    double fbrem_mode =  bestGsfElectron.fbrem();
    h1_ele_fbrem->Fill(fbrem_mode);

    if (bestGsfElectron.isEB())
     {
      double fbrem_mode_barrel = bestGsfElectron.fbrem();
      h1_ele_fbrem_barrel->Fill(fbrem_mode_barrel);
     }

    if (bestGsfElectron.isEE())
     {
      double fbrem_mode_endcaps = bestGsfElectron.fbrem();
      h1_ele_fbrem_endcaps->Fill(fbrem_mode_endcaps);
     }

    double superclusterfbrem_mode =  bestGsfElectron.superClusterFbrem();
    h1_ele_superclusterfbrem->Fill(superclusterfbrem_mode);

    if (bestGsfElectron.isEB())
     {
      double superclusterfbrem_mode_barrel = bestGsfElectron.superClusterFbrem();
      h1_ele_superclusterfbrem_barrel->Fill(superclusterfbrem_mode_barrel);
     }

    if (bestGsfElectron.isEE())
     {
      double superclusterfbrem_mode_endcaps = bestGsfElectron.superClusterFbrem();
      h1_ele_superclusterfbrem_endcaps->Fill(superclusterfbrem_mode_endcaps);
     }

    p1_ele_fbremVsEta_mode->Fill(bestGsfElectron.eta(),fbrem_mode);

    if (!readAOD_) // track extra does not exist in AOD
     {
      double fbrem_mean =  1. - bestGsfElectron.gsfTrack()->outerMomentum().R()/bestGsfElectron.gsfTrack()->innerMomentum().R() ;
      p1_ele_fbremVsEta_mean->Fill(bestGsfElectron.eta(),fbrem_mean) ;
     }

    //

    if (bestGsfElectron.classification() == GsfElectron::GOLDEN) h2_ele_PinVsPoutGolden_mode->Fill(bestGsfElectron.trackMomentumOut().R(), bestGsfElectron.trackMomentumAtVtx().R());
    if (bestGsfElectron.classification() == GsfElectron::SHOWERING) h2_ele_PinVsPoutShowering_mode->Fill(bestGsfElectron.trackMomentumOut().R(), bestGsfElectron.trackMomentumAtVtx().R());
    if (!readAOD_) // track extra not available in AOD
     {
      if (bestGsfElectron.classification() == GsfElectron::GOLDEN) h2_ele_PinVsPoutGolden_mean->Fill(bestGsfElectron.gsfTrack()->outerMomentum().R(), bestGsfElectron.gsfTrack()->innerMomentum().R());
      if (bestGsfElectron.classification() == GsfElectron::SHOWERING) h2_ele_PinVsPoutShowering_mean->Fill(bestGsfElectron.gsfTrack()->outerMomentum().R(), bestGsfElectron.gsfTrack()->innerMomentum().R());
     }
    if (bestGsfElectron.classification() == GsfElectron::GOLDEN) h2_ele_PtinVsPtoutGolden_mode->Fill(bestGsfElectron.trackMomentumOut().Rho(), bestGsfElectron.trackMomentumAtVtx().Rho());
    if (bestGsfElectron.classification() == GsfElectron::SHOWERING) h2_ele_PtinVsPtoutShowering_mode->Fill(bestGsfElectron.trackMomentumOut().Rho(), bestGsfElectron.trackMomentumAtVtx().Rho());
    if (!readAOD_) // track extra not available in AOD
     {
      if (bestGsfElectron.classification() == GsfElectron::GOLDEN) h2_ele_PtinVsPtoutGolden_mean->Fill(bestGsfElectron.gsfTrack()->outerMomentum().Rho(), bestGsfElectron.gsfTrack()->innerMomentum().Rho());
      if (bestGsfElectron.classification() == GsfElectron::SHOWERING) h2_ele_PtinVsPtoutShowering_mean->Fill(bestGsfElectron.gsfTrack()->outerMomentum().Rho(), bestGsfElectron.gsfTrack()->innerMomentum().Rho());
     }

    // provenance and pflow data
    h1_ele_mva->Fill(bestGsfElectron.mva_e_pi());
    if (bestGsfElectron.isEB()) h1_ele_mva_barrel->Fill(bestGsfElectron.mva_e_pi());
    if (bestGsfElectron.isEE()) h1_ele_mva_endcaps->Fill(bestGsfElectron.mva_e_pi());
    h1_ele_mva_isolated->Fill(bestGsfElectron.mva_Isolated());
    if (bestGsfElectron.isEB()) h1_ele_mva_barrel_isolated->Fill(bestGsfElectron.mva_Isolated());
    if (bestGsfElectron.isEE()) h1_ele_mva_endcaps_isolated->Fill(bestGsfElectron.mva_Isolated());
    if (bestGsfElectron.ecalDrivenSeed()) h1_ele_provenance->Fill(1.);
    if (bestGsfElectron.trackerDrivenSeed()) h1_ele_provenance->Fill(-1.);
    if (bestGsfElectron.trackerDrivenSeed()||bestGsfElectron.ecalDrivenSeed()) h1_ele_provenance->Fill(0.);
    if (bestGsfElectron.trackerDrivenSeed()&&!bestGsfElectron.ecalDrivenSeed()) h1_ele_provenance->Fill(-2.);
    if (!bestGsfElectron.trackerDrivenSeed()&&bestGsfElectron.ecalDrivenSeed()) h1_ele_provenance->Fill(2.);

    if (bestGsfElectron.ecalDrivenSeed() && bestGsfElectron.isEB()) h1_ele_provenance_barrel->Fill(1.);
    if (bestGsfElectron.trackerDrivenSeed() && bestGsfElectron.isEB()) h1_ele_provenance_barrel->Fill(-1.);
    if ((bestGsfElectron.trackerDrivenSeed()||bestGsfElectron.ecalDrivenSeed()) && bestGsfElectron.isEB()) h1_ele_provenance_barrel->Fill(0.);
    if (bestGsfElectron.trackerDrivenSeed()&&!bestGsfElectron.ecalDrivenSeed() && bestGsfElectron.isEB()) h1_ele_provenance_barrel->Fill(-2.);
    if (!bestGsfElectron.trackerDrivenSeed()&&bestGsfElectron.ecalDrivenSeed() && bestGsfElectron.isEB()) h1_ele_provenance_barrel->Fill(2.);
    if (bestGsfElectron.ecalDrivenSeed() && bestGsfElectron.isEE()) h1_ele_provenance_endcaps->Fill(1.);
    if (bestGsfElectron.trackerDrivenSeed() && bestGsfElectron.isEE()) h1_ele_provenance_endcaps->Fill(-1.);
    if ((bestGsfElectron.trackerDrivenSeed()||bestGsfElectron.ecalDrivenSeed()) && bestGsfElectron.isEE()) h1_ele_provenance_endcaps->Fill(0.);
    if (bestGsfElectron.trackerDrivenSeed()&&!bestGsfElectron.ecalDrivenSeed() && bestGsfElectron.isEE()) h1_ele_provenance_endcaps->Fill(-2.);
    if (!bestGsfElectron.trackerDrivenSeed()&&bestGsfElectron.ecalDrivenSeed() && bestGsfElectron.isEE()) h1_ele_provenance_endcaps->Fill(2.);

    if (bestGsfElectron.isGsfCtfScPixChargeConsistent()) h1_ele_chargeInfo->Fill(-1.0);
    if (bestGsfElectron.isGsfScPixChargeConsistent()) h1_ele_chargeInfo->Fill(0.);
    if (bestGsfElectron.isGsfCtfChargeConsistent()) h1_ele_chargeInfo->Fill(1.0);

    // Pflow isolation
    h1_ele_chargedHadronIso->Fill(bestGsfElectron.pfIsolationVariables().sumChargedHadronPt);
    if (bestGsfElectron.isEB()) h1_ele_chargedHadronIso_barrel->Fill(bestGsfElectron.pfIsolationVariables().sumChargedHadronPt);
    if (bestGsfElectron.isEE()) h1_ele_chargedHadronIso_endcaps->Fill(bestGsfElectron.pfIsolationVariables().sumChargedHadronPt);

    h1_ele_neutralHadronIso->Fill(bestGsfElectron.pfIsolationVariables().sumNeutralHadronEt);
    if (bestGsfElectron.isEB()) h1_ele_neutralHadronIso_barrel->Fill(bestGsfElectron.pfIsolationVariables().sumNeutralHadronEt);
    if (bestGsfElectron.isEE()) h1_ele_neutralHadronIso_endcaps->Fill(bestGsfElectron.pfIsolationVariables().sumNeutralHadronEt);

    h1_ele_photonIso->Fill(bestGsfElectron.pfIsolationVariables().sumPhotonEt);
    if (bestGsfElectron.isEB()) h1_ele_photonIso_barrel->Fill(bestGsfElectron.pfIsolationVariables().sumPhotonEt);
    if (bestGsfElectron.isEE()) h1_ele_photonIso_endcaps->Fill(bestGsfElectron.pfIsolationVariables().sumPhotonEt);

	// -- pflow over pT
	h1_ele_chargedHadronRelativeIso->Fill(bestGsfElectron.pfIsolationVariables().sumChargedHadronPt / bestGsfElectron.pt());
	if (bestGsfElectron.isEB()) h1_ele_chargedHadronRelativeIso_barrel->Fill(bestGsfElectron.pfIsolationVariables().sumChargedHadronPt / bestGsfElectron.pt());
	if (bestGsfElectron.isEE()) h1_ele_chargedHadronRelativeIso_endcaps->Fill(bestGsfElectron.pfIsolationVariables().sumChargedHadronPt / bestGsfElectron.pt());

    h1_ele_neutralHadronRelativeIso->Fill(bestGsfElectron.pfIsolationVariables().sumNeutralHadronEt / bestGsfElectron.pt());
    if (bestGsfElectron.isEB()) h1_ele_neutralHadronRelativeIso_barrel->Fill(bestGsfElectron.pfIsolationVariables().sumNeutralHadronEt / bestGsfElectron.pt());
    if (bestGsfElectron.isEE()) h1_ele_neutralHadronRelativeIso_endcaps->Fill(bestGsfElectron.pfIsolationVariables().sumNeutralHadronEt / bestGsfElectron.pt());

    h1_ele_photonRelativeIso->Fill(bestGsfElectron.pfIsolationVariables().sumPhotonEt / bestGsfElectron.pt());
    if (bestGsfElectron.isEB()) h1_ele_photonRelativeIso_barrel->Fill(bestGsfElectron.pfIsolationVariables().sumPhotonEt / bestGsfElectron.pt());
    if (bestGsfElectron.isEE()) h1_ele_photonRelativeIso_endcaps->Fill(bestGsfElectron.pfIsolationVariables().sumPhotonEt / bestGsfElectron.pt());

    // isolation
    h1_ele_tkSumPt_dr03->Fill(bestGsfElectron.dr03TkSumPt());
    if (bestGsfElectron.isEB()) h1_ele_tkSumPt_dr03_barrel->Fill(bestGsfElectron.dr03TkSumPt());
    if (bestGsfElectron.isEE()) h1_ele_tkSumPt_dr03_endcaps->Fill(bestGsfElectron.dr03TkSumPt());
    h1_ele_ecalRecHitSumEt_dr03->Fill(bestGsfElectron.dr03EcalRecHitSumEt());
    if (bestGsfElectron.isEB()) h1_ele_ecalRecHitSumEt_dr03_barrel->Fill(bestGsfElectron.dr03EcalRecHitSumEt());
    if (bestGsfElectron.isEE()) h1_ele_ecalRecHitSumEt_dr03_endcaps->Fill(bestGsfElectron.dr03EcalRecHitSumEt());
    h1_ele_hcalTowerSumEt_dr03_depth1->Fill(bestGsfElectron.dr03HcalDepth1TowerSumEt());
    if (bestGsfElectron.isEB()) h1_ele_hcalTowerSumEt_dr03_depth1_barrel->Fill(bestGsfElectron.dr03HcalDepth1TowerSumEt());
    if (bestGsfElectron.isEE()) h1_ele_hcalTowerSumEt_dr03_depth1_endcaps->Fill(bestGsfElectron.dr03HcalDepth1TowerSumEt());
    h1_ele_hcalTowerSumEt_dr03_depth2->Fill(bestGsfElectron.dr03HcalDepth2TowerSumEt());
    if (bestGsfElectron.isEB()) h1_ele_hcalTowerSumEt_dr03_depth2_barrel->Fill(bestGsfElectron.dr03HcalDepth2TowerSumEt());
    if (bestGsfElectron.isEE()) h1_ele_hcalTowerSumEt_dr03_depth2_endcaps->Fill(bestGsfElectron.dr03HcalDepth2TowerSumEt());
    h1_ele_hcalTowerSumEtBc_dr03_depth1->Fill(bestGsfElectron.dr03HcalDepth1TowerSumEtBc());
    if (bestGsfElectron.isEB()) h1_ele_hcalTowerSumEtBc_dr03_depth1_barrel->Fill(bestGsfElectron.dr03HcalDepth1TowerSumEtBc());
    if (bestGsfElectron.isEE()) h1_ele_hcalTowerSumEtBc_dr03_depth1_endcaps->Fill(bestGsfElectron.dr03HcalDepth1TowerSumEtBc());
    h1_ele_hcalTowerSumEtBc_dr03_depth2->Fill(bestGsfElectron.dr03HcalDepth2TowerSumEtBc());
    if (bestGsfElectron.isEB()) h1_ele_hcalTowerSumEtBc_dr03_depth2_barrel->Fill(bestGsfElectron.dr03HcalDepth2TowerSumEtBc());
    if (bestGsfElectron.isEE()) h1_ele_hcalTowerSumEtBc_dr03_depth2_endcaps->Fill(bestGsfElectron.dr03HcalDepth2TowerSumEtBc());
    h1_ele_tkSumPt_dr04->Fill(bestGsfElectron.dr04TkSumPt());
    if (bestGsfElectron.isEB()) h1_ele_tkSumPt_dr04_barrel->Fill(bestGsfElectron.dr04TkSumPt());
    if (bestGsfElectron.isEE()) h1_ele_tkSumPt_dr04_endcaps->Fill(bestGsfElectron.dr04TkSumPt());
    h1_ele_ecalRecHitSumEt_dr04->Fill(bestGsfElectron.dr04EcalRecHitSumEt());
    if (bestGsfElectron.isEB()) h1_ele_ecalRecHitSumEt_dr04_barrel->Fill(bestGsfElectron.dr04EcalRecHitSumEt());
    if (bestGsfElectron.isEE()) h1_ele_ecalRecHitSumEt_dr04_endcaps->Fill(bestGsfElectron.dr04EcalRecHitSumEt());
    h1_ele_hcalTowerSumEt_dr04_depth1->Fill(bestGsfElectron.dr04HcalDepth1TowerSumEt());
    if (bestGsfElectron.isEB()) h1_ele_hcalTowerSumEt_dr04_depth1_barrel->Fill(bestGsfElectron.dr04HcalDepth1TowerSumEt());
    if (bestGsfElectron.isEE()) h1_ele_hcalTowerSumEt_dr04_depth1_endcaps->Fill(bestGsfElectron.dr04HcalDepth1TowerSumEt());
    h1_ele_hcalTowerSumEt_dr04_depth2->Fill(bestGsfElectron.dr04HcalDepth2TowerSumEt());
    if (bestGsfElectron.isEB()) h1_ele_hcalTowerSumEt_dr04_depth2_barrel->Fill(bestGsfElectron.dr04HcalDepth2TowerSumEt());
    if (bestGsfElectron.isEE()) h1_ele_hcalTowerSumEt_dr04_depth2_endcaps->Fill(bestGsfElectron.dr04HcalDepth2TowerSumEt());

    h1_ele_hcalTowerSumEtBc_dr04_depth1->Fill(bestGsfElectron.dr04HcalDepth1TowerSumEtBc());
    if (bestGsfElectron.isEB()) h1_ele_hcalTowerSumEtBc_dr04_depth1_barrel->Fill(bestGsfElectron.dr04HcalDepth1TowerSumEtBc());
    if (bestGsfElectron.isEE()) h1_ele_hcalTowerSumEtBc_dr04_depth1_endcaps->Fill(bestGsfElectron.dr04HcalDepth1TowerSumEtBc());
    h1_ele_hcalTowerSumEtBc_dr04_depth2->Fill(bestGsfElectron.dr04HcalDepth2TowerSumEtBc());
    if (bestGsfElectron.isEB()) h1_ele_hcalTowerSumEtBc_dr04_depth2_barrel->Fill(bestGsfElectron.dr04HcalDepth2TowerSumEtBc());
    if (bestGsfElectron.isEE()) h1_ele_hcalTowerSumEtBc_dr04_depth2_endcaps->Fill(bestGsfElectron.dr04HcalDepth2TowerSumEtBc());

    h1_ele_hcalDepth1OverEcalBc->Fill(bestGsfElectron.hcalDepth1OverEcalBc());
    if (bestGsfElectron.isEB()) h1_ele_hcalDepth1OverEcalBc_barrel->Fill(bestGsfElectron.hcalDepth1OverEcalBc());
    if (bestGsfElectron.isEE()) h1_ele_hcalDepth1OverEcalBc_endcaps->Fill(bestGsfElectron.hcalDepth1OverEcalBc());
    h1_ele_hcalDepth2OverEcalBc->Fill(bestGsfElectron.hcalDepth2OverEcalBc());
    if (bestGsfElectron.isEB()) h1_ele_hcalDepth2OverEcalBc_barrel->Fill(bestGsfElectron.hcalDepth2OverEcalBc());
    if (bestGsfElectron.isEE()) h1_ele_hcalDepth2OverEcalBc_endcaps->Fill(bestGsfElectron.hcalDepth2OverEcalBc());

    // conversion rejection
    int flags = bestGsfElectron.convFlags() ;
    if (flags==-9999) { flags=-1 ; }
    h1_ele_convFlags->Fill(flags);
    if (flags>=0.)
     {
      h1_ele_convDist->Fill( bestGsfElectron.convDist() );
      h1_ele_convDcot->Fill( bestGsfElectron.convDcot() );
      h1_ele_convRadius->Fill( bestGsfElectron.convRadius() );
     }

   } // loop over mc particle
  h1_mcNum->Fill(mcNum) ;
  h1_eleNum->Fill(eleNum) ;
 }


