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
#include "SimTracker/TrackAssociation/interface/TrackAssociatorBase.h"
#include "MagneticField/Engine/interface/MagneticField.h"
//
//DQM services
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/MonitorElement.h"

//
#include <map>
#include <vector>
/** \class PhotonValidator
 **  
 **
 **  $Id: PhotonValidator
 **  $Date: 2009/05/04 11:48:57 $ 
 **  $Revision: 1.17 $
 **  \author Nancy Marinelli, U. of Notre Dame, US
 **
 ***/


// forward declarations
class TFile;
class TH1F;
class TH2F;
class TProfile;
class TTree;
class SimVertex;
class SimTrack;


class PhotonValidator : public edm::EDAnalyzer
{

 public:
   
  //
  explicit PhotonValidator( const edm::ParameterSet& ) ;
  virtual ~PhotonValidator();
                                   
      
  virtual void analyze( const edm::Event&, const edm::EventSetup& ) ;
  virtual void beginJob();
  virtual void beginRun( edm::Run const & r, edm::EventSetup const & theEventSetup) ;
  virtual void endJob() ;
  
 private:
  //

  float  phiNormalization( float& a);
  float  etaTransformation( float a, float b);
  std::vector<float>  errors( TH1* histo1, TH1* histo2 );  
  void fillPlotFromVectors(MonitorElement* h, std::vector<int>& numerator, std::vector<int>& denominator,std::string type);
  void initVectors();
  void doProfileX(TH2 * th2, MonitorElement* me);
  void doProfileX(MonitorElement * th2m, MonitorElement* me);

      
  std::string fName_;
  DQMStore *dbe_;
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

  edm::InputTag  bcBarrelCollection_;
  edm::InputTag  bcEndcapCollection_;
 
  edm::InputTag barrelEcalHits_;
  edm::InputTag endcapEcalHits_;

  edm::InputTag label_tp_;


  std::string conversionOITrackProducer_;
  std::string conversionIOTrackProducer_;



  PhotonMCTruthFinder*  thePhotonMCTruthFinder_;
  TrackAssociatorBase * theTrackAssociator_;

  double minPhoEtCut_;
  double convTrackMinPtCut_;
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
  bool signal_;

  /// global variable for the MC photon
  double mcPhi_;
  double mcEta_;
  double mcConvR_;      
  double mcConvZ_;
  double mcConvY_;            
  double mcConvX_;            
  std::vector<TrackingParticleRef> theConvTP_;
  
  double simMinPt_;
  double simMaxPt_;
  
  /// Global variables for reco Photon
  double recMinPt_;
  double recMaxPt_;

  std::vector<double> etaintervals_, etaintervalslarge_, phiintervals_, rintervals_, zintervals_;
  std::vector<double> etintervals_;
  std::vector<int> totSimPhoEta_, totSimPhoEt_,  totMatchedSimPhoEta_, totMatchedSimPhoEt_, totMatchedSimPhoEtaSmallR9_, totSimPhoPhi_, totMatchedSimPhoPhi_;
  std::vector<int> totSimConvEta_, totSimConvPhi_, totSimConvR_, totSimConvZ_, totSimConvEt_;
  std::vector<int> totMatchedSimConvEtaTwoTracks_,totMatchedSimConvEtaAllTwoTracks_,totMatchedSimConvPhiTwoTracks_, totMatchedSimConvRTwoTracks_,totMatchedSimConvRAllTwoTracks_, totMatchedSimConvZTwoTracks_,totMatchedSimConvEtTwoTracks_ ;
  std::vector<int> totMatchedConvEtaTwoTracksWithVtx_,totMatchedConvEtaTwoTracksWithVtxProbGT0_,totMatchedConvEtaTwoTracksWithVtxProbGT005_,totMatchedConvEtaTwoTracksWithVtxProbGT01_;
  std::vector<int> totMatchedSimConvEtaOneTrack_, totMatchedSimConvPhiOneTrack_,  totMatchedSimConvROneTrack_, totMatchedSimConvZOneTrack_, totMatchedSimConvEtOneTrack_ ;
  std::vector<int> totMatchedRecConvEtaTwoTracks_,totMatchedRecConvPhiTwoTracks_, totMatchedRecConvRTwoTracks_, totMatchedRecConvZTwoTracks_,totMatchedRecConvEtTwoTracks_ ;
  std::vector<int> totRecAssConvEtaTwoTracks_,totRecAssConvPhiTwoTracks_, totRecAssConvRTwoTracks_, totRecAssConvZTwoTracks_,totRecAssConvEtTwoTracks_ ;

  //
  MonitorElement* h_nSimPho_[2];
  MonitorElement* h_SimPhoE_[2];
  MonitorElement* h_SimPhoEt_[2];
  MonitorElement* h_SimPhoPhi_[2];
  MonitorElement* h_SimPhoEta_[2];
  MonitorElement* h_SimPhoMotherType_[2];
  MonitorElement* h_SimPhoMotherEt_[2];
  MonitorElement* h_SimPhoMotherEta_[2];
  MonitorElement* h_SimPhoEtaSmallR9_;
  //
  MonitorElement* h_nSimConv_[2];
  MonitorElement* h_SimConvE_[2];
  MonitorElement* h_SimConvEt_[2];
  MonitorElement* h_SimConvPhi_[2];
  MonitorElement* h_SimConvEta_[2];
  MonitorElement* h_SimConvEtaPix_[2];

  MonitorElement* h_SimConvR_[2];
  MonitorElement* h_SimConvZ_[2];
  //
  MonitorElement* h_simTkPt_;
  MonitorElement* h_simTkEta_;


  MonitorElement*  phoEffEta_;
  MonitorElement*  phoEffPhi_;
  MonitorElement*  phoEffEt_;


  MonitorElement*  convEffEtaTwoTracks_;
  MonitorElement*  convEffEtaTwoTracksR9_;
  MonitorElement*  convEffPhiTwoTracks_;
  MonitorElement*  convEffRTwoTracks_;
  MonitorElement*  convEffRAllTwoTracks_;
  MonitorElement*  convEffZTwoTracks_;
  MonitorElement*  convEffEtTwoTracks_;
  MonitorElement*  convEffEtaTwoTracksAndVtx_;
  MonitorElement*  convEffEtaTwoTracksAndVtx2_;
  MonitorElement*  convEffEtaTwoTracksAndVtxProbGT0_;
  MonitorElement*  convEffEtaTwoTracksAndVtxProbGT005_;
  MonitorElement*  convEffEtaTwoTracksAndVtxProbGT01_;
  MonitorElement*  convEffEtaAllTwoTracks_;


  MonitorElement*  convFakeRateEtaTwoTracks_;
  MonitorElement*  convFakeRatePhiTwoTracks_;
  MonitorElement*  convFakeRateRTwoTracks_;
  MonitorElement*  convFakeRateZTwoTracks_;
  MonitorElement*  convFakeRateEtTwoTracks_;


  MonitorElement*  convEffEtaOneTrack_;
  MonitorElement*  convEffPhiOneTrack_;
  MonitorElement*  convEffROneTrack_;
  MonitorElement*  convEffZOneTrack_;
  MonitorElement*  convEffEtOneTrack_;

  //// test on OutIn Tracks
  MonitorElement* h_OIinnermostHitR_;
  MonitorElement* h_IOinnermostHitR_;
  MonitorElement* h_trkProv_[2];


  MonitorElement* h_phoDEta_[2];
  MonitorElement* h_phoDPhi_[2];


  MonitorElement* h_nPho_[2][3];
  MonitorElement* h_scEta_[2];
  MonitorElement* h_scEtaWidth_[2];
  MonitorElement* h_scPhi_[2];
  MonitorElement* h_scPhiWidth_[2];
  MonitorElement* h_scEtaPhi_[2];

 
  MonitorElement* h_scE_[2][3];
  MonitorElement* h_scEt_[2][3];

  MonitorElement* h_psE_;

  MonitorElement* h_r9_[3][3];  
  MonitorElement* h2_r9VsEta_[3];
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

  MonitorElement* h_phoE_[2][3];
  MonitorElement* h_phoEt_[2][3];
  MonitorElement* h_phoERes_[3][3];

  MonitorElement* h2_eResVsEta_[3];
  MonitorElement* p_eResVsEta_[3];


  MonitorElement* h_phoEta_[2];
  MonitorElement* h_phoPhi_[2];



  /// info per conversion
  MonitorElement* h_nConv_[2][3];
  MonitorElement* h_convEta_[3];
  MonitorElement* h_convPhi_[2];
  MonitorElement* h_convERes_[2][3];
  MonitorElement* h_convPRes_[2][3];

  MonitorElement* h_invMass_[2][3];
  MonitorElement* h_r9VsNofTracks_[2][3];
  MonitorElement* h_EoverPTracks_[2][3];
  MonitorElement* h_PoverETracks_[2][3];


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
  MonitorElement* h_convVtxdX_;
  MonitorElement* h_convVtxdY_;
  MonitorElement* h_convVtxdZ_;
  MonitorElement* h_convVtxdR_;

  MonitorElement* h2_convVtxdRVsR_;
  MonitorElement* p_convVtxdRVsR_;
  MonitorElement* h2_convVtxdRVsEta_;
  MonitorElement* p_convVtxdRVsEta_;

  MonitorElement* h2_convVtxRrecVsTrue_;

  MonitorElement*  h_vtxChi2_[3];
  MonitorElement*  h_vtxChi2Prob_[3];



  MonitorElement* h_zPVFromTracks_[2]; 
  MonitorElement* h_dzPVFromTracks_[2]; 
  MonitorElement* h2_dzPVVsR_;
  MonitorElement* p_dzPVVsR_;


  //////////// info per track
  MonitorElement* h_nHitsVsEta_[2]; 
  MonitorElement* nHitsVsEta_[2]; 
  MonitorElement* h_nHitsVsR_[2]; 
  MonitorElement* nHitsVsR_[2]; 
  MonitorElement* h_tkChi2_[2];
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

  //
  //
  


};

#endif
