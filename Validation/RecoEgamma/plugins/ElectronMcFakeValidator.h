
#ifndef Validation_RecoEgamma_ElectronMcFakeValidator_h
#define Validation_RecoEgamma_ElectronMcFakeValidator_h

#include "DQMOffline/EGamma/interface/ElectronDqmAnalyzerBase.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
class MagneticField;

#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronCore.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedFwd.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/ValueMap.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

class ElectronMcFakeValidator : public ElectronDqmAnalyzerBase {
public:
  explicit ElectronMcFakeValidator(const edm::ParameterSet &conf);
  ~ElectronMcFakeValidator() override;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void analyze(const edm::Event &e, const edm::EventSetup &c) override;

private:
  edm::EDGetTokenT<reco::GsfElectronCollection> electronCollection_;
  edm::EDGetTokenT<reco::GsfElectronCoreCollection> electronCoreCollection_;
  edm::EDGetTokenT<reco::GsfTrackCollection> electronTrackCollection_;
  edm::EDGetTokenT<reco::ElectronSeedCollection> electronSeedCollection_;
  edm::EDGetTokenT<reco::GenJetCollection> matchingObjectCollection_;
  edm::EDGetTokenT<reco::VertexCollection> offlineVerticesCollection_;  // new 2015.04.02
  edm::EDGetTokenT<reco::BeamSpot> beamSpotTag_;
  bool readAOD_;

  edm::EDGetTokenT<edm::ValueMap<double>> isoFromDepsTk03Tag_;
  edm::EDGetTokenT<edm::ValueMap<double>> isoFromDepsTk04Tag_;
  edm::EDGetTokenT<edm::ValueMap<double>> isoFromDepsEcalFull03Tag_;
  edm::EDGetTokenT<edm::ValueMap<double>> isoFromDepsEcalFull04Tag_;
  edm::EDGetTokenT<edm::ValueMap<double>> isoFromDepsEcalReduced03Tag_;
  edm::EDGetTokenT<edm::ValueMap<double>> isoFromDepsEcalReduced04Tag_;
  edm::EDGetTokenT<edm::ValueMap<double>> isoFromDepsHcal03Tag_;
  edm::EDGetTokenT<edm::ValueMap<double>> isoFromDepsHcal04Tag_;

  edm::ESHandle<TrackerGeometry> pDD;
  edm::ESHandle<MagneticField> theMagField;

  float mcEnergy[10], mcEta[10], mcPhi[10], mcPt[10], mcQ[10];
  float superclusterEnergy[10], superclusterEta[10], superclusterPhi[10], superclusterEt[10];
  float seedMomentum[10], seedEta[10], seedPhi[10], seedPt[10], seedQ[10];

  double maxPt_;
  double maxAbsEta_;
  double deltaR_;

  std::string inputFile_;
  std::string outputFile_;
  std::string inputInternalPath_;
  std::string outputInternalPath_;

  // histos limits and binning

  int xyz_nbin;
  int p_nbin;
  int p2D_nbin;
  double p_max;
  int pt_nbin;
  int pt2D_nbin;
  int pteff_nbin;
  double pt_max;
  int fhits_nbin;
  double fhits_max;
  int lhits_nbin;
  double lhits_max;
  int eta_nbin;
  int eta2D_nbin;
  double eta_min;
  double eta_max;
  int deta_nbin;
  double deta_min;
  double deta_max;
  int detamatch_nbin;
  int detamatch2D_nbin;
  double detamatch_min;
  double detamatch_max;
  int phi_nbin;
  int phi2D_nbin;
  double phi_min;
  double phi_max;
  int dphi_nbin;
  double dphi_min;
  double dphi_max;
  int dphimatch_nbin;
  int dphimatch2D_nbin;
  double dphimatch_min;
  double dphimatch_max;
  int eop_nbin;
  int eop2D_nbin;
  double eop_max;
  double eopmaxsht;
  int mee_nbin;
  double mee_min;
  double mee_max;
  int hoe_nbin;
  double hoe_min;
  double hoe_max;
  int popmatching_nbin;
  double popmatching_min;
  double popmatching_max;
  bool set_EfficiencyFlag;
  bool set_StatOverflowFlag;
  int opv_nbin;
  double opv_min;
  double opv_max;
  int ele_nbin;
  double ele_min;
  double ele_max;
  int core_nbin;
  double core_min;
  double core_max;
  int track_nbin;
  double track_min;
  double track_max;
  int seed_nbin;
  double seed_min;
  double seed_max;

  // histos

  MonitorElement *h1_matchingObjectNum;
  MonitorElement *h1_recEleNum_;
  MonitorElement *h1_recCoreNum_;
  MonitorElement *h1_recTrackNum_;
  MonitorElement *h1_recSeedNum_;

  MonitorElement *h1_recOfflineVertices_;  // new 2015.04.02

  MonitorElement *h1_matchingObjectEta;
  MonitorElement *h1_matchingObjectAbsEta;
  MonitorElement *h1_matchingObjectP;
  MonitorElement *h1_matchingObjectPt;
  MonitorElement *h1_matchingObjectPhi;
  MonitorElement *h1_matchingObjectZ;

  MonitorElement *h1_ele_EoverP_all;
  MonitorElement *h1_ele_EseedOP_all;
  MonitorElement *h1_ele_EoPout_all;
  MonitorElement *h1_ele_EeleOPout_all;
  MonitorElement *h1_ele_dEtaSc_propVtx_all;
  MonitorElement *h1_ele_dPhiSc_propVtx_all;
  MonitorElement *h1_ele_dEtaCl_propOut_all;
  MonitorElement *h1_ele_dPhiCl_propOut_all;
  MonitorElement *h1_ele_TIP_all;
  MonitorElement *h1_ele_HoE_all;
  MonitorElement *h1_ele_HoE_bc_all;
  MonitorElement *h1_ele_vertexEta_all;
  MonitorElement *h1_ele_vertexPt_all;
  MonitorElement *h1_ele_mee_all;
  MonitorElement *h1_ele_mee_os;

  MonitorElement *h2_ele_E2mnE1vsMee_all;
  MonitorElement *h2_ele_E2mnE1vsMee_egeg_all;

  MonitorElement *h1_ele_matchingObjectEta_matched;
  MonitorElement *h1_ele_matchingObjectAbsEta_matched;
  MonitorElement *h1_ele_matchingObjectPt_matched;
  MonitorElement *h1_ele_matchingObjectPhi_matched;
  MonitorElement *h1_ele_matchingObjectZ_matched;

  MonitorElement *h1_ele_charge;
  MonitorElement *h2_ele_chargeVsEta;
  MonitorElement *h2_ele_chargeVsPhi;
  MonitorElement *h2_ele_chargeVsPt;
  MonitorElement *h1_ele_vertexP;
  MonitorElement *h1_ele_vertexPt;
  MonitorElement *h2_ele_vertexPtVsEta;
  MonitorElement *h2_ele_vertexPtVsPhi;
  MonitorElement *h1_ele_vertexEta;
  MonitorElement *h2_ele_vertexEtaVsPhi;
  MonitorElement *h1_ele_vertexAbsEta;
  MonitorElement *h1_ele_vertexPhi;
  MonitorElement *h1_ele_vertexX;
  MonitorElement *h1_ele_vertexY;
  MonitorElement *h1_ele_vertexZ;
  MonitorElement *h1_ele_vertexTIP;
  MonitorElement *h2_ele_vertexTIPVsEta;
  MonitorElement *h2_ele_vertexTIPVsPhi;
  MonitorElement *h2_ele_vertexTIPVsPt;

  MonitorElement *h1_ele_PoPmatchingObject;
  MonitorElement *h2_ele_PoPmatchingObjectVsEta;
  MonitorElement *h2_ele_PoPmatchingObjectVsPhi;
  MonitorElement *h2_ele_PoPmatchingObjectVsPt;
  MonitorElement *h1_ele_PoPmatchingObject_barrel;
  MonitorElement *h1_ele_PoPmatchingObject_endcaps;

  MonitorElement *h1_ele_EtaMnEtamatchingObject;
  MonitorElement *h2_ele_EtaMnEtamatchingObjectVsEta;
  MonitorElement *h2_ele_EtaMnEtamatchingObjectVsPhi;
  MonitorElement *h2_ele_EtaMnEtamatchingObjectVsPt;
  MonitorElement *h1_ele_PhiMnPhimatchingObject;
  MonitorElement *h1_ele_PhiMnPhimatchingObject2;
  MonitorElement *h2_ele_PhiMnPhimatchingObjectVsEta;
  MonitorElement *h2_ele_PhiMnPhimatchingObjectVsPhi;
  MonitorElement *h2_ele_PhiMnPhimatchingObjectVsPt;

  MonitorElement *h1_scl_En_;
  MonitorElement *h1_scl_EoEmatchingObject_barrel;
  MonitorElement *h1_scl_EoEmatchingObject_endcaps;
  MonitorElement *h1_scl_Et_;
  MonitorElement *h2_scl_EtVsEta_;
  MonitorElement *h2_scl_EtVsPhi_;
  MonitorElement *h2_scl_EtaVsPhi_;
  MonitorElement *h1_scl_Eta_;
  MonitorElement *h1_scl_Phi_;

  MonitorElement *h1_scl_SigIEtaIEta_;
  MonitorElement *h1_scl_SigIEtaIEta_barrel_;
  MonitorElement *h1_scl_SigIEtaIEta_endcaps_;
  MonitorElement *h1_scl_full5x5_sigmaIetaIeta_;          // new 2014.01.12
  MonitorElement *h1_scl_full5x5_sigmaIetaIeta_barrel_;   // new 2014.01.12
  MonitorElement *h1_scl_full5x5_sigmaIetaIeta_endcaps_;  // new 2014.01.12
  MonitorElement *h1_scl_E1x5_;
  MonitorElement *h1_scl_E1x5_barrel_;
  MonitorElement *h1_scl_E1x5_endcaps_;
  MonitorElement *h1_scl_E2x5max_;
  MonitorElement *h1_scl_E2x5max_barrel_;
  MonitorElement *h1_scl_E2x5max_endcaps_;
  MonitorElement *h1_scl_E5x5_;
  MonitorElement *h1_scl_E5x5_barrel_;
  MonitorElement *h1_scl_E5x5_endcaps_;

  MonitorElement *h1_ele_ambiguousTracks;
  MonitorElement *h2_ele_ambiguousTracksVsEta;
  MonitorElement *h2_ele_ambiguousTracksVsPhi;
  MonitorElement *h2_ele_ambiguousTracksVsPt;
  MonitorElement *h1_ele_foundHits;
  MonitorElement *h1_ele_foundHits_barrel;
  MonitorElement *h1_ele_foundHits_endcaps;
  MonitorElement *h2_ele_foundHitsVsEta;
  MonitorElement *h2_ele_foundHitsVsPhi;
  MonitorElement *h2_ele_foundHitsVsPt;
  MonitorElement *h1_ele_lostHits;
  MonitorElement *h1_ele_lostHits_barrel;
  MonitorElement *h1_ele_lostHits_endcaps;
  MonitorElement *h2_ele_lostHitsVsEta;
  MonitorElement *h2_ele_lostHitsVsPhi;
  MonitorElement *h2_ele_lostHitsVsPt;
  MonitorElement *h1_ele_chi2;
  MonitorElement *h1_ele_chi2_barrel;
  MonitorElement *h1_ele_chi2_endcaps;
  MonitorElement *h2_ele_chi2VsEta;
  MonitorElement *h2_ele_chi2VsPhi;
  MonitorElement *h2_ele_chi2VsPt;

  MonitorElement *h1_ele_PinMnPout;
  MonitorElement *h1_ele_PinMnPout_mode;
  MonitorElement *h2_ele_PinMnPoutVsEta_mode;
  MonitorElement *h2_ele_PinMnPoutVsPhi_mode;
  MonitorElement *h2_ele_PinMnPoutVsPt_mode;
  MonitorElement *h2_ele_PinMnPoutVsE_mode;
  MonitorElement *h2_ele_PinMnPoutVsChi2_mode;

  MonitorElement *h1_ele_outerP;
  MonitorElement *h1_ele_outerP_mode;
  MonitorElement *h2_ele_outerPVsEta_mode;
  MonitorElement *h1_ele_outerPt;
  MonitorElement *h1_ele_outerPt_mode;
  MonitorElement *h2_ele_outerPtVsEta_mode;
  MonitorElement *h2_ele_outerPtVsPhi_mode;
  MonitorElement *h2_ele_outerPtVsPt_mode;
  MonitorElement *h1_ele_EoP;
  MonitorElement *h1_ele_EoP_barrel;
  MonitorElement *h1_ele_EoP_endcaps;
  MonitorElement *h2_ele_EoPVsEta;
  MonitorElement *h2_ele_EoPVsPhi;
  MonitorElement *h2_ele_EoPVsE;
  MonitorElement *h1_ele_EseedOP;
  MonitorElement *h1_ele_EseedOP_barrel;
  MonitorElement *h1_ele_EseedOP_endcaps;
  MonitorElement *h2_ele_EseedOPVsEta;
  MonitorElement *h2_ele_EseedOPVsPhi;
  MonitorElement *h2_ele_EseedOPVsE;
  MonitorElement *h1_ele_EoPout;
  MonitorElement *h1_ele_EoPout_barrel;
  MonitorElement *h1_ele_EoPout_endcaps;
  MonitorElement *h2_ele_EoPoutVsEta;
  MonitorElement *h2_ele_EoPoutVsPhi;
  MonitorElement *h2_ele_EoPoutVsE;
  MonitorElement *h1_ele_EeleOPout;
  MonitorElement *h1_ele_EeleOPout_barrel;
  MonitorElement *h1_ele_EeleOPout_endcaps;
  MonitorElement *h2_ele_EeleOPoutVsEta;
  MonitorElement *h2_ele_EeleOPoutVsPhi;
  MonitorElement *h2_ele_EeleOPoutVsE;

  MonitorElement *h1_ele_dEtaSc_propVtx;
  MonitorElement *h1_ele_dEtaSc_propVtx_barrel;
  MonitorElement *h1_ele_dEtaSc_propVtx_endcaps;
  MonitorElement *h2_ele_dEtaScVsEta_propVtx;
  MonitorElement *h2_ele_dEtaScVsPhi_propVtx;
  MonitorElement *h2_ele_dEtaScVsPt_propVtx;
  MonitorElement *h1_ele_dPhiSc_propVtx;
  MonitorElement *h1_ele_dPhiSc_propVtx_barrel;
  MonitorElement *h1_ele_dPhiSc_propVtx_endcaps;
  MonitorElement *h2_ele_dPhiScVsEta_propVtx;
  MonitorElement *h2_ele_dPhiScVsPhi_propVtx;
  MonitorElement *h2_ele_dPhiScVsPt_propVtx;
  MonitorElement *h1_ele_dEtaCl_propOut;
  MonitorElement *h1_ele_dEtaCl_propOut_barrel;
  MonitorElement *h1_ele_dEtaCl_propOut_endcaps;
  MonitorElement *h2_ele_dEtaClVsEta_propOut;
  MonitorElement *h2_ele_dEtaClVsPhi_propOut;
  MonitorElement *h2_ele_dEtaClVsPt_propOut;
  MonitorElement *h1_ele_dPhiCl_propOut;
  MonitorElement *h1_ele_dPhiCl_propOut_barrel;
  MonitorElement *h1_ele_dPhiCl_propOut_endcaps;
  MonitorElement *h2_ele_dPhiClVsEta_propOut;
  MonitorElement *h2_ele_dPhiClVsPhi_propOut;
  MonitorElement *h2_ele_dPhiClVsPt_propOut;
  MonitorElement *h1_ele_dEtaEleCl_propOut;
  MonitorElement *h1_ele_dEtaEleCl_propOut_barrel;
  MonitorElement *h1_ele_dEtaEleCl_propOut_endcaps;
  MonitorElement *h2_ele_dEtaEleClVsEta_propOut;
  MonitorElement *h2_ele_dEtaEleClVsPhi_propOut;
  MonitorElement *h2_ele_dEtaEleClVsPt_propOut;
  MonitorElement *h1_ele_dPhiEleCl_propOut;
  MonitorElement *h1_ele_dPhiEleCl_propOut_barrel;
  MonitorElement *h1_ele_dPhiEleCl_propOut_endcaps;
  MonitorElement *h2_ele_dPhiEleClVsEta_propOut;
  MonitorElement *h2_ele_dPhiEleClVsPhi_propOut;
  MonitorElement *h2_ele_dPhiEleClVsPt_propOut;

  MonitorElement *h1_ele_seed_subdet2_;
  MonitorElement *h1_ele_seed_mask_;
  MonitorElement *h1_ele_seed_mask_bpix_;
  MonitorElement *h1_ele_seed_mask_fpix_;
  MonitorElement *h1_ele_seed_mask_tec_;
  MonitorElement *h1_ele_seed_dphi2_;
  MonitorElement *h2_ele_seed_dphi2VsEta_;
  MonitorElement *h2_ele_seed_dphi2VsPt_;
  MonitorElement *h1_ele_seed_dphi2pos_;
  MonitorElement *h2_ele_seed_dphi2posVsEta_;
  MonitorElement *h2_ele_seed_dphi2posVsPt_;
  MonitorElement *h1_ele_seed_drz2_;
  MonitorElement *h2_ele_seed_drz2VsEta_;
  MonitorElement *h2_ele_seed_drz2VsPt_;
  MonitorElement *h1_ele_seed_drz2pos_;
  MonitorElement *h2_ele_seed_drz2posVsEta_;
  MonitorElement *h2_ele_seed_drz2posVsPt_;

  MonitorElement *h1_ele_classes;
  MonitorElement *h1_ele_eta;
  MonitorElement *h1_ele_eta_golden;
  MonitorElement *h1_ele_eta_bbrem;
  MonitorElement *h1_ele_eta_narrow;
  MonitorElement *h1_ele_eta_shower;

  MonitorElement *h1_ele_HoE;
  MonitorElement *h1_ele_HoE_bc;
  MonitorElement *h1_ele_HoE_barrel;
  MonitorElement *h1_ele_HoE_endcaps;
  MonitorElement *h1_ele_HoE_bc_barrel;
  MonitorElement *h1_ele_HoE_bc_endcaps;
  MonitorElement *h1_ele_HoE_fiducial;
  MonitorElement *h2_ele_HoEVsEta;
  MonitorElement *h2_ele_HoEVsPhi;
  MonitorElement *h2_ele_HoEVsE;
  //    MonitorElement *h1_scl_ESFrac;
  MonitorElement *h1_scl_ESFrac_endcaps;

  MonitorElement *h1_ele_fbrem;
  MonitorElement *h1_ele_fbrem_barrel;
  MonitorElement *h1_ele_fbrem_endcaps;
  MonitorElement *h1_ele_superclusterfbrem;
  MonitorElement *h1_ele_superclusterfbrem_barrel;
  MonitorElement *h1_ele_superclusterfbrem_endcaps;
  MonitorElement *p1_ele_fbremVsEta_mode;
  MonitorElement *p1_ele_fbremVsEta_mean;

  MonitorElement *h2_ele_PinVsPoutGolden_mode;
  MonitorElement *h2_ele_PinVsPoutShowering_mode;
  MonitorElement *h2_ele_PinVsPoutGolden_mean;
  MonitorElement *h2_ele_PinVsPoutShowering_mean;
  MonitorElement *h2_ele_PtinVsPtoutGolden_mode;
  MonitorElement *h2_ele_PtinVsPtoutShowering_mode;
  MonitorElement *h2_ele_PtinVsPtoutGolden_mean;
  MonitorElement *h2_ele_PtinVsPtoutShowering_mean;
  MonitorElement *h1_scl_EoEmatchingObjectGolden_barrel;
  MonitorElement *h1_scl_EoEmatchingObjectGolden_endcaps;
  MonitorElement *h1_scl_EoEmatchingObjectShowering_barrel;
  MonitorElement *h1_scl_EoEmatchingObjectShowering_endcaps;

  MonitorElement *h1_ele_mva;
  MonitorElement *h1_ele_mva_barrel;
  MonitorElement *h1_ele_mva_endcaps;
  MonitorElement *h1_ele_mva_isolated;
  MonitorElement *h1_ele_mva_barrel_isolated;
  MonitorElement *h1_ele_mva_endcaps_isolated;
  MonitorElement *h1_ele_provenance;
  MonitorElement *h1_ele_provenance_barrel;
  MonitorElement *h1_ele_provenance_endcaps;

  // pflow isolation
  MonitorElement *h1_ele_chargedHadronIso;
  MonitorElement *h1_ele_chargedHadronIso_barrel;
  MonitorElement *h1_ele_chargedHadronIso_endcaps;
  MonitorElement *h1_ele_neutralHadronIso;
  MonitorElement *h1_ele_neutralHadronIso_barrel;
  MonitorElement *h1_ele_neutralHadronIso_endcaps;
  MonitorElement *h1_ele_photonIso;
  MonitorElement *h1_ele_photonIso_barrel;
  MonitorElement *h1_ele_photonIso_endcaps;
  // -- pflow over pT
  MonitorElement *h1_ele_chargedHadronRelativeIso;
  MonitorElement *h1_ele_chargedHadronRelativeIso_barrel;
  MonitorElement *h1_ele_chargedHadronRelativeIso_endcaps;
  MonitorElement *h1_ele_neutralHadronRelativeIso;
  MonitorElement *h1_ele_neutralHadronRelativeIso_barrel;
  MonitorElement *h1_ele_neutralHadronRelativeIso_endcaps;
  MonitorElement *h1_ele_photonRelativeIso;
  MonitorElement *h1_ele_photonRelativeIso_barrel;
  MonitorElement *h1_ele_photonRelativeIso_endcaps;

  MonitorElement *h1_ele_tkSumPt_dr03;
  MonitorElement *h1_ele_tkSumPt_dr03_barrel;
  MonitorElement *h1_ele_tkSumPt_dr03_endcaps;
  MonitorElement *h1_ele_ecalRecHitSumEt_dr03;
  MonitorElement *h1_ele_ecalRecHitSumEt_dr03_barrel;
  MonitorElement *h1_ele_ecalRecHitSumEt_dr03_endcaps;
  MonitorElement *h1_ele_hcalTowerSumEt_dr03_depth1;
  MonitorElement *h1_ele_hcalTowerSumEt_dr03_depth1_barrel;
  MonitorElement *h1_ele_hcalTowerSumEt_dr03_depth1_endcaps;
  MonitorElement *h1_ele_hcalTowerSumEt_dr03_depth2;
  MonitorElement *h1_ele_tkSumPt_dr04;
  MonitorElement *h1_ele_tkSumPt_dr04_barrel;
  MonitorElement *h1_ele_tkSumPt_dr04_endcaps;
  MonitorElement *h1_ele_ecalRecHitSumEt_dr04;
  MonitorElement *h1_ele_ecalRecHitSumEt_dr04_barrel;
  MonitorElement *h1_ele_ecalRecHitSumEt_dr04_endcaps;
  MonitorElement *h1_ele_hcalTowerSumEt_dr04_depth1;
  MonitorElement *h1_ele_hcalTowerSumEt_dr04_depth1_barrel;
  MonitorElement *h1_ele_hcalTowerSumEt_dr04_depth1_endcaps;
  MonitorElement *h1_ele_hcalTowerSumEt_dr04_depth2;

  MonitorElement *h1_ele_hcalDepth1OverEcalBc;
  MonitorElement *h1_ele_hcalDepth1OverEcalBc_barrel;
  MonitorElement *h1_ele_hcalDepth1OverEcalBc_endcaps;
  MonitorElement *h1_ele_hcalDepth2OverEcalBc;
  MonitorElement *h1_ele_hcalDepth2OverEcalBc_barrel;
  MonitorElement *h1_ele_hcalDepth2OverEcalBc_endcaps;

  MonitorElement *h1_ele_hcalTowerSumEtBc_dr03_depth1;
  MonitorElement *h1_ele_hcalTowerSumEtBc_dr03_depth1_barrel;
  MonitorElement *h1_ele_hcalTowerSumEtBc_dr03_depth1_endcaps;
  MonitorElement *h1_ele_hcalTowerSumEtBc_dr03_depth2;
  MonitorElement *h1_ele_hcalTowerSumEtBc_dr03_depth2_barrel;
  MonitorElement *h1_ele_hcalTowerSumEtBc_dr03_depth2_endcaps;
  MonitorElement *h1_ele_hcalTowerSumEtBc_dr04_depth1;
  MonitorElement *h1_ele_hcalTowerSumEtBc_dr04_depth1_barrel;
  MonitorElement *h1_ele_hcalTowerSumEtBc_dr04_depth1_endcaps;
  MonitorElement *h1_ele_hcalTowerSumEtBc_dr04_depth2;
  MonitorElement *h1_ele_hcalTowerSumEtBc_dr04_depth2_barrel;
  MonitorElement *h1_ele_hcalTowerSumEtBc_dr04_depth2_endcaps;

  MonitorElement *h1_ele_convFlags;
  MonitorElement *h1_ele_convFlags_all;
  MonitorElement *h1_ele_convDist;
  MonitorElement *h1_ele_convDist_all;
  MonitorElement *h1_ele_convDcot;
  MonitorElement *h1_ele_convDcot_all;
  MonitorElement *h1_ele_convRadius;
  MonitorElement *h1_ele_convRadius_all;
};

#endif
