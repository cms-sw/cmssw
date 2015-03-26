#ifndef PhotonValidator_H
#define PhotonValidator_H
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "RecoEgamma/EgammaMCTools/interface/PhotonMCTruthFinder.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
//#include "RecoEgamma/EgammaTools/interface/ConversionLikelihoodCalculator.h"
//
//DQM services
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"


//
#include <map>
#include <vector>
#include <memory>
/** \class PhotonValidator
 **
 **
 **  $Id: PhotonValidator
 **  \author Nancy Marinelli, U. of Notre Dame, US
 **
 ***/


// forward declarations
namespace edm {class HepMCProduct;}
class TFile;
class TH1F;
class TH2F;
class TProfile;
class TTree;
class SimVertex;
class SimTrack;



class PhotonValidator : public DQMEDAnalyzer
{

 public:

  //
  explicit PhotonValidator( const edm::ParameterSet& ) ;
  virtual ~PhotonValidator();


  virtual void analyze( const edm::Event&, const edm::EventSetup& ) override;
  //  virtual void beginJob();
  virtual void dqmBeginRun( edm::Run const & r, edm::EventSetup const & theEventSetup) override;
  virtual void endRun (edm::Run const& r, edm::EventSetup const & es) override;
  void  bookHistograms( DQMStore::IBooker&, edm::Run const &, edm::EventSetup const &) override; 

 private:
  //

  float  phiNormalization( float& a);
  float  etaTransformation( float a, float b);


  std::string fName_;
  edm::ESHandle<MagneticField> theMF_;

  int verbosity_;
  int nEvt_;
  int nEntry_;
  int nSimPho_[2];
  int nSimConv_[2];
  int nMatched_;
  int nRecConv_;
  int nRecConvAss_;
  int nRecConvAssWithEcal_;

  int nInvalidPCA_;

  edm::ParameterSet parameters_;
  edm::ESHandle<CaloGeometry> theCaloGeom_;
  edm::ESHandle<CaloTopology> theCaloTopo_;


  std::string photonCollectionProducer_;
  std::string photonCollection_;
  edm::EDGetTokenT<reco::PhotonCollection> photonCollectionToken_;
  edm::EDGetTokenT<reco::PFCandidateCollection> pfCandidates_;
  std::string   valueMapPhoPFCandIso_ ;
  edm::EDGetTokenT<edm::ValueMap<std::vector<reco::PFCandidateRef> > > particleBasedIso_token;
  edm::EDGetTokenT<reco::VertexCollection> offline_pvToken_;
  edm::InputTag  bcBarrelCollection_;
  edm::InputTag  bcEndcapCollection_;


  edm::EDGetTokenT<EcalRecHitCollection> barrelEcalHits_;
  edm::EDGetTokenT<EcalRecHitCollection> endcapEcalHits_;
  edm::EDGetTokenT<TrackingParticleCollection> token_tp_;

 

  edm::InputTag conversionOITrackProducer_;
  edm::InputTag conversionIOTrackProducer_;

  edm::EDGetTokenT<edm::View<reco::Track> > conversionOITrackPr_Token_;
  edm::EDGetTokenT<edm::View<reco::Track> > conversionIOTrackPr_Token_;

  edm::EDGetTokenT<edm::SimTrackContainer>  g4_simTk_Token_;
  edm::EDGetTokenT<edm::SimVertexContainer> g4_simVtx_Token_;
  edm::EDGetTokenT<edm::SimTrackContainer>  famos_simTk_Token_;
  edm::EDGetTokenT<edm::SimVertexContainer> famos_simVtx_Token_;
  edm::EDGetTokenT<edm::HepMCProduct>  hepMC_Token_;
  edm::EDGetTokenT<reco::GenJetCollection> genjets_Token_;

  std::unique_ptr<PhotonMCTruthFinder>  thePhotonMCTruthFinder_;

  bool fastSim_;
  bool isRunCentrally_;


  double minPhoEtCut_;
  double convTrackMinPtCut_;
  double likelihoodCut_;
  double trkIsolExtRadius_;
  double trkIsolInnRadius_;
  double trkPtLow_;
  double lip_;
  double ecalIsolRadius_;
  double bcEtLow_;
  double hcalIsolExtRadius_;
  double hcalIsolInnRadius_;
  double hcalHitEtLow_;
  int  numOfTracksInCone_;
  double trkPtSumCut_;
  double ecalEtSumCut_;
  double hcalEtSumCut_;
  bool dCotCutOn_;
  double dCotCutValue_;
  double dCotHardCutValue_;


  /// global variable for the MC photon
  double mcPhi_;
  double mcEta_;
  double mcConvR_;
  double mcConvZ_;
  double mcConvY_;
  double mcConvX_;
  double mcConvPhi_;
  double mcConvEta_;
  double mcJetEta_;
  double mcJetPhi_;

  edm::RefVector<TrackingParticleCollection> theConvTP_;
 //  std::vector<TrackingParticleRef> theConvTP_;

  double simMinPt_;
  double simMaxPt_;

  /// Global variables for reco Photon
  double recMinPt_;
  double recMaxPt_;
  MonitorElement* h_nRecoVtx_;
  //
  MonitorElement* h_nSimPho_[2];
  MonitorElement* h_SimPhoMotherType_[2];
  MonitorElement* h_SimPhoMotherEt_[2];
  MonitorElement* h_SimPhoMotherEta_[2];
  MonitorElement* h_SimPhoEtaSmallR9_;
  //
  MonitorElement* h_nSimConv_[2];
  MonitorElement* h_SimConvEtaPix_[2];
  //
  MonitorElement* h_simTkPt_;
  MonitorElement* h_simTkEta_;

  MonitorElement* h_simConvVtxRvsZ_[4];
  MonitorElement* h_simConvVtxYvsX_;


  ///   Denominator for efficiencies
  MonitorElement*   h_SimPho_[3];
  MonitorElement*   h_AllSimConv_[5];
  MonitorElement*   h_VisSimConv_[6];
  MonitorElement*   h_VisSimConvLarge_;
  ///   Numerator for efficiencies
  MonitorElement*   h_MatchedSimPho_[3];
  MonitorElement*   h_MatchedSimPhoBadCh_[3];
  MonitorElement*   h_SimConvOneTracks_[5];
  MonitorElement*   h_SimConvOneMTracks_[5];
  MonitorElement*   h_SimConvTwoTracks_[5];
  MonitorElement*   h_SimConvTwoMTracks_[5];
  MonitorElement*   h_SimConvMTotal_[5];
  MonitorElement*   h_SimConvTwoMTracksAndVtxPGT0_[5];
  MonitorElement*   h_SimConvTwoMTracksAndVtxPGT0005_[5];
  MonitorElement*   h_SimConvTwoMTracksAndVtxPGT01_[5];
  // Denominators for conversion fake rate
  MonitorElement*   h_RecoConvTwoTracks_[5];
  // Numerators for conversion fake rate
  MonitorElement*   h_RecoConvTwoMTracks_[5];

  //// tmp TH1F
  TH1F* th1f_SimConvMTotal_[5];
 
  //// test on OutIn Tracks
  MonitorElement* h_OIinnermostHitR_;
  MonitorElement* h_IOinnermostHitR_;
  MonitorElement* h_trkProv_[2];
  MonitorElement* h_trkAlgo_;
  MonitorElement* h_convAlgo_;
  MonitorElement* h_convQuality_;


  MonitorElement* h_phoDEta_[2];
  MonitorElement* h_phoDPhi_[2];



  MonitorElement* h_scEta_[2];
  MonitorElement* h_scEtaWidth_[2];
  MonitorElement* h_scPhi_[2];
  MonitorElement* h_scPhiWidth_[2];
  MonitorElement* h_scEtaPhi_[2];


  MonitorElement* h_scE_[2][3];
  MonitorElement* h_scEt_[2][3];

  MonitorElement* h_psE_;

  MonitorElement* h_EtR9Less093_[3][3];
  MonitorElement* h_r9_[3][3];
  MonitorElement* p_r9VsEta_[3];
  MonitorElement* h2_r9VsEt_[3];
  MonitorElement* p_r9VsEt_[3];
  //
  MonitorElement* h_r1_[3][3];
  MonitorElement* h2_r1VsEta_[3];
  MonitorElement* p_r1VsEta_[3];
  MonitorElement* h2_r1VsEt_[3];
  MonitorElement* p_r1VsEt_[3];
  //
  MonitorElement* h_r2_[3][3];
  MonitorElement* h2_r2VsEta_[3];
  MonitorElement* p_r2VsEta_[3];
  MonitorElement* h2_r2VsEt_[3];
  MonitorElement* p_r2VsEt_[3];
  //
  MonitorElement* h_sigmaIetaIeta_[3][3];
  MonitorElement* h2_sigmaIetaIetaVsEta_[3];
  MonitorElement* p_sigmaIetaIetaVsEta_[3];
  MonitorElement* h2_sigmaIetaIetaVsEt_[3];
  MonitorElement* p_sigmaIetaIetaVsEt_[3];
  //
  MonitorElement* h_hOverE_[3][3];
  MonitorElement* h2_hOverEVsEta_[3];
  MonitorElement* p_hOverEVsEta_[3];
  MonitorElement* h2_hOverEVsEt_[3];
  MonitorElement* p_hOverEVsEt_[3];
  //
  MonitorElement* h_newhOverE_[3][3];
  MonitorElement* p_newhOverEVsEta_[3];
  MonitorElement* p_newhOverEVsEt_[3];





  //
  MonitorElement* h_ecalRecHitSumEtConeDR04_[3][3];
  MonitorElement* h2_ecalRecHitSumEtConeDR04VsEta_[3];
  MonitorElement* p_ecalRecHitSumEtConeDR04VsEta_[3];
  MonitorElement* h2_ecalRecHitSumEtConeDR04VsEt_[3];
  MonitorElement* p_ecalRecHitSumEtConeDR04VsEt_[3];
  //
  MonitorElement* h_hcalTowerSumEtConeDR04_[3][3];
  MonitorElement* h2_hcalTowerSumEtConeDR04VsEta_[3];
  MonitorElement* p_hcalTowerSumEtConeDR04VsEta_[3];
  MonitorElement* h2_hcalTowerSumEtConeDR04VsEt_[3];
  MonitorElement* p_hcalTowerSumEtConeDR04VsEt_[3];
  //
  MonitorElement* h_hcalTowerBcSumEtConeDR04_[3][3];
  MonitorElement* p_hcalTowerBcSumEtConeDR04VsEta_[3];
  MonitorElement* p_hcalTowerBcSumEtConeDR04VsEt_[3];
  //
  MonitorElement* h_isoTrkSolidConeDR04_[3][3];
  MonitorElement* h2_isoTrkSolidConeDR04VsEta_[3];
  MonitorElement* p_isoTrkSolidConeDR04VsEta_[3];
  MonitorElement* h2_isoTrkSolidConeDR04VsEt_[3];
  MonitorElement* p_isoTrkSolidConeDR04VsEt_[3];
  //
  MonitorElement* h_nTrkSolidConeDR04_[3][3];
  MonitorElement* h2_nTrkSolidConeDR04VsEta_[3];
  MonitorElement* p_nTrkSolidConeDR04VsEta_[3];
  MonitorElement* h2_nTrkSolidConeDR04VsEt_[3];
  MonitorElement* p_nTrkSolidConeDR04VsEt_[3];
  //

  MonitorElement*  h_gamgamMass_[3][3];
  MonitorElement*  h_gamgamMassRegr1_[3][3];
  MonitorElement*  h_gamgamMassRegr2_[3][3];

  MonitorElement* h_phoE_[2][3];
  MonitorElement* h_phoEt_[2][3];
  MonitorElement* h_phoERes_[3][3];
  MonitorElement* h_phoSigmaEoE_[3][3];


  MonitorElement* h2_eResVsEta_[3];
  MonitorElement* p_eResVsEta_[3];
  MonitorElement* p_sigmaEoEVsEta_[3];
  MonitorElement* h2_eResVsEt_[3][3];
  MonitorElement* p_eResVsEt_[3][3];
  MonitorElement* p_eResVsNVtx_[3][3];

  MonitorElement* p_sigmaEoEVsEt_[3][3];
  MonitorElement* p_sigmaEoEVsNVtx_[3][3];

  MonitorElement* h2_eResVsR9_[3];
  MonitorElement* p_eResVsR9_[3];
  MonitorElement* h2_sceResVsR9_[3];
  MonitorElement* p_sceResVsR9_[3];

  MonitorElement* h_phoEta_[2];
  MonitorElement* h_phoPhi_[2];

  // Photon energies as derived from Regression1 (MIT) nd Regression2 (PF/Rishi)
  MonitorElement* h_phoEResRegr1_[3][3];
  MonitorElement* h_phoEResRegr2_[3][3];

  // 
  MonitorElement* h_phoPixSeedSize_[2];

  // Information from Particle Flow
  // Isolation
  MonitorElement* h_chHadIso_[3];
  MonitorElement* h_nHadIso_[3];
  MonitorElement* h_phoIso_[3];
  // Identification
  MonitorElement* h_nCluOutsideMustache_[3];
  MonitorElement* h_etOutsideMustache_[3];
  MonitorElement* h_pfMva_[3];
  //// particle based isolation from ValueMap
  MonitorElement* h_dRPhoPFcand_ChHad_Cleaned_[3];
  MonitorElement* h_dRPhoPFcand_NeuHad_Cleaned_[3];
  MonitorElement* h_dRPhoPFcand_Pho_Cleaned_[3];
  MonitorElement* h_dRPhoPFcand_ChHad_unCleaned_[3];
  MonitorElement* h_dRPhoPFcand_NeuHad_unCleaned_[3];
  MonitorElement* h_dRPhoPFcand_Pho_unCleaned_[3];
  MonitorElement* h_SumPtOverPhoPt_ChHad_Cleaned_[3];
  MonitorElement* h_SumPtOverPhoPt_NeuHad_Cleaned_[3];
  MonitorElement* h_SumPtOverPhoPt_Pho_Cleaned_[3];
  MonitorElement* h_SumPtOverPhoPt_ChHad_unCleaned_[3];
  MonitorElement* h_SumPtOverPhoPt_NeuHad_unCleaned_[3];
  MonitorElement* h_SumPtOverPhoPt_Pho_unCleaned_[3];


  /// info per conversion
  MonitorElement* h_nConv_[2][3];
  MonitorElement* h_convEta_[3];
  MonitorElement* h_convPhi_[2];
  MonitorElement* h_convERes_[2][3];
  MonitorElement*  p_eResVsR_;

  MonitorElement* h_convPtRes_[2][3];

  MonitorElement* h_invMass_[2][3];
  MonitorElement* h_r9VsNofTracks_[2][3];
  MonitorElement* h_EoverPTracks_[2][3];
  MonitorElement* h_PoverETracks_[2][3];

  MonitorElement* h_EoverP_SL_[3];

  MonitorElement* h_mvaOut_[3];
  MonitorElement* h2_etaVsRsim_[3];
  MonitorElement* h2_etaVsRreco_[3];

  MonitorElement* h2_EoverEtrueVsEoverP_[3];
  MonitorElement* h2_PoverPtrueVsEoverP_[3];

  MonitorElement* h2_EoverPVsEta_[3];
  MonitorElement* p_EoverPVsEta_[3];
  MonitorElement* h2_EoverPVsR_[3];
  MonitorElement* p_EoverPVsR_[3];

  MonitorElement* h2_EoverEtrueVsEta_[3];
  MonitorElement* p_EoverEtrueVsEta_[3];
  MonitorElement* h2_EoverEtrueVsR_[3];
  MonitorElement* p_EoverEtrueVsR_[3];


  MonitorElement* h2_PoverPtrueVsEta_[3];
  MonitorElement* p_PoverPtrueVsEta_[3];

  MonitorElement* h_DPhiTracksAtVtx_[2][3];
  MonitorElement* h2_DPhiTracksAtVtxVsEta_;
  MonitorElement* p_DPhiTracksAtVtxVsEta_;
  MonitorElement* h2_DPhiTracksAtVtxVsR_;
  MonitorElement* p_DPhiTracksAtVtxVsR_;

  MonitorElement* h_DCotTracks_[2][3];
  MonitorElement* h2_DCotTracksVsEta_;
  MonitorElement* p_DCotTracksVsEta_;
  MonitorElement* h2_DCotTracksVsR_;
  MonitorElement* p_DCotTracksVsR_;

  MonitorElement* h_distMinAppTracks_[2][3];



  MonitorElement* h_DPhiTracksAtEcal_[2][3];
  MonitorElement* h2_DPhiTracksAtEcalVsR_;
  MonitorElement* p_DPhiTracksAtEcalVsR_;
  MonitorElement* h2_DPhiTracksAtEcalVsEta_;
  MonitorElement* p_DPhiTracksAtEcalVsEta_;


  MonitorElement* h_DEtaTracksAtEcal_[2][3];



  MonitorElement* h_convVtxRvsZ_[3];
  MonitorElement* h_convVtxYvsX_;
  MonitorElement* h_convVtxRvsZ_zoom_[2];
  MonitorElement* h_convVtxYvsX_zoom_[2];
  MonitorElement* h_convSLVtxRvsZ_[3];

  MonitorElement* h_convVtxdX_;
  MonitorElement* h_convVtxdY_;
  MonitorElement* h_convVtxdZ_;
  MonitorElement* h_convVtxdR_;

  MonitorElement* h_convVtxdX_barrel_;
  MonitorElement* h_convVtxdY_barrel_;
  MonitorElement* h_convVtxdZ_barrel_;
  MonitorElement* h_convVtxdR_barrel_;

  MonitorElement* h_convVtxdX_endcap_;
  MonitorElement* h_convVtxdY_endcap_;
  MonitorElement* h_convVtxdZ_endcap_;
  MonitorElement* h_convVtxdR_endcap_;

  MonitorElement* h_convVtxdEta_;
  MonitorElement* h_convVtxdPhi_;


  MonitorElement* h2_convVtxdRVsR_;
  MonitorElement* p_convVtxdRVsR_;
  MonitorElement* h2_convVtxdRVsEta_;
  MonitorElement* p_convVtxdRVsEta_;
  MonitorElement* p_convVtxdXVsX_;
  MonitorElement* p_convVtxdYVsY_;
  MonitorElement* p_convVtxdZVsZ_;

  MonitorElement* h2_convVtxRrecVsTrue_;

  MonitorElement*  h_vtxChi2_[3];
  MonitorElement*  h_vtxChi2Prob_[3];



  MonitorElement* h_zPVFromTracks_[5];
  MonitorElement* h_dzPVFromTracks_[5];
  MonitorElement* h2_dzPVVsR_;
  MonitorElement* p_dzPVVsR_;
  MonitorElement* p_dzPVVsEta_;


  //////////// info per track
  MonitorElement* p_nHitsVsEta_[2];
  MonitorElement* p_nHitsVsEtaSL_[2];
  MonitorElement* nHitsVsEta_[2];
  MonitorElement* p_nHitsVsR_[2];
  MonitorElement* p_nHitsVsRSL_[2];
  MonitorElement* nHitsVsR_[2];
  MonitorElement* h_tkChi2_[2];
  MonitorElement* h_tkChi2SL_[2];
  MonitorElement* h_tkChi2Large_[2];
  MonitorElement* h2_Chi2VsEta_[3];
  MonitorElement* p_Chi2VsEta_[3];
  MonitorElement* h2_Chi2VsR_[3];
  MonitorElement* p_Chi2VsR_[3];

  MonitorElement* h_TkD0_[3];

  MonitorElement* h_TkPtPull_[3];
  MonitorElement* h2_TkPtPull_[3];
  MonitorElement* p_TkPtPull_[3];
  MonitorElement* h2_PtRecVsPtSim_[3];
  MonitorElement* h2_PtRecVsPtSimMixProv_;

  MonitorElement* hBCEnergyOverTrackPout_[3];

  // ME for bkg efficiencies
  MonitorElement*   h_SimJet_[3];
  MonitorElement*   h_MatchedSimJet_[3];
  MonitorElement*   h_MatchedSimJetBadCh_[3];
  //

  MonitorElement*   h_nPho_;

  MonitorElement* h_scBkgEta_;
  MonitorElement* h_scBkgPhi_;
  MonitorElement* h_phoBkgEta_;
  MonitorElement* h_phoBkgPhi_;
  MonitorElement* h_phoBkgDEta_;
  MonitorElement* h_phoBkgDPhi_;
  MonitorElement* h_phoBkgE_[3];
  MonitorElement* h_phoBkgEt_[3];


  MonitorElement* h_scBkgE_[3];
  MonitorElement* h_scBkgEt_[3];

  MonitorElement* h_r9Bkg_[3];
  MonitorElement* h_r1Bkg_[3];
  MonitorElement* h_r2Bkg_[3];
  MonitorElement* h_hOverEBkg_[3];

  MonitorElement* h2_r9VsEtaBkg_;
  MonitorElement* h2_r9VsEtBkg_;

  MonitorElement* h2_r1VsEtaBkg_;
  MonitorElement* h2_r1VsEtBkg_;
  MonitorElement* p_r1VsEtaBkg_;
  MonitorElement* p_r1VsEtBkg_;

  MonitorElement* h2_r2VsEtaBkg_;
  MonitorElement* h2_r2VsEtBkg_;
  MonitorElement* p_r2VsEtaBkg_;
  MonitorElement* p_r2VsEtBkg_;

  MonitorElement* h_sigmaIetaIetaBkg_[3];
  MonitorElement* h2_sigmaIetaIetaVsEtaBkg_;
  MonitorElement* p_sigmaIetaIetaVsEtaBkg_;
  MonitorElement* h2_sigmaIetaIetaVsEtBkg_[3];
  MonitorElement* p_sigmaIetaIetaVsEtBkg_[3];


  MonitorElement* h2_hOverEVsEtaBkg_;
  MonitorElement* h2_hOverEVsEtBkg_;
  MonitorElement* p_hOverEVsEtaBkg_;
  MonitorElement* p_hOverEVsEtBkg_;


  MonitorElement* h_ecalRecHitSumEtConeDR04Bkg_[3];
  MonitorElement* h2_ecalRecHitSumEtConeDR04VsEtaBkg_;
  MonitorElement* p_ecalRecHitSumEtConeDR04VsEtaBkg_;
  MonitorElement* h2_ecalRecHitSumEtConeDR04VsEtBkg_[3];
  MonitorElement* p_ecalRecHitSumEtConeDR04VsEtBkg_[3];


  MonitorElement* h_hcalTowerSumEtConeDR04Bkg_[3];
  MonitorElement* h2_hcalTowerSumEtConeDR04VsEtaBkg_;
  MonitorElement* p_hcalTowerSumEtConeDR04VsEtaBkg_;
  MonitorElement* h2_hcalTowerSumEtConeDR04VsEtBkg_[3];
  MonitorElement* p_hcalTowerSumEtConeDR04VsEtBkg_[3];

  MonitorElement* h_isoTrkSolidConeDR04Bkg_[3];
  MonitorElement* h2_isoTrkSolidConeDR04VsEtaBkg_;
  MonitorElement* p_isoTrkSolidConeDR04VsEtaBkg_;
  MonitorElement* h2_isoTrkSolidConeDR04VsEtBkg_[3];
  MonitorElement* p_isoTrkSolidConeDR04VsEtBkg_[3];
  //
  MonitorElement* h_nTrkSolidConeDR04Bkg_[3];
  MonitorElement* h2_nTrkSolidConeDR04VsEtaBkg_;
  MonitorElement* p_nTrkSolidConeDR04VsEtaBkg_;
  MonitorElement* h2_nTrkSolidConeDR04VsEtBkg_[3];
  MonitorElement* p_nTrkSolidConeDR04VsEtBkg_[3];
  //
  MonitorElement* h_convEtaBkg_;
  MonitorElement* h_convPhiBkg_;
  MonitorElement* h_mvaOutBkg_[3];
  MonitorElement* nHitsVsEtaBkg_;
  MonitorElement* h_tkChi2Bkg_;
  MonitorElement* h_EoverPTracksBkg_[3];
  MonitorElement* h_PoverETracksBkg_[3];
  MonitorElement* h_DPhiTracksAtVtxBkg_[3];
  MonitorElement* h_DCotTracksBkg_[3];
  MonitorElement* h_convVtxYvsXBkg_;
  MonitorElement* h_convVtxRvsZBkg_[2];



class  sortPhotons
{
  public:
    bool operator () (const reco::PhotonRef& lhs, const reco::PhotonRef & rhs)
    {
        return lhs->et() > rhs->et();
    }
};



};







#endif
