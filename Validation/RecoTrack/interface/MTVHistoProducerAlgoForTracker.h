#ifndef Validation_RecoTrack_MTVHistoProducerAlgoForTracker_h
#define Validation_RecoTrack_MTVHistoProducerAlgoForTracker_h

/* \author B.Mangano, UCSD
 *
 * Concrete class implementing the MTVHistoProducerAlgo interface.
 * To be used within the MTV to fill histograms for Tracker tracks.
 */

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Validation/RecoTrack/interface/MTVHistoProducerAlgo.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "CommonTools/RecoAlgos/interface/TrackingParticleSelector.h"

#include "DQMServices/Core/interface/MonitorElement.h"
#include <TH1F.h>
#include <TH2F.h>



class MTVHistoProducerAlgoForTracker: public MTVHistoProducerAlgo {
 public:
  MTVHistoProducerAlgoForTracker(const edm::ParameterSet& pset) ;
  virtual ~MTVHistoProducerAlgoForTracker();

  void initialize(){setUpVectors();};

  void bookSimHistos();

  void bookRecoHistos();
  void bookRecoHistosForStandaloneRunning();


  void fill_generic_simTrack_histos(int counter,ParticleBase::Vector,ParticleBase::Point vertex);


  void fill_recoAssociated_simTrack_histos(int count,
					   const TrackingParticle& tp,
					   ParticleBase::Vector momentumTP,ParticleBase::Point vertexTP,
					   double dxy, double dz, int nSimHits,
					   const reco::Track* track);


  void fill_generic_recoTrack_histos(int count,
				     const reco::Track& track,
				     math::XYZPoint bsPosition,
				     bool isMatched);

  void fill_dedx_recoTrack_histos(int count, edm::RefToBase<reco::Track>& trackref, std::vector< edm::ValueMap<reco::DeDxData> > v_dEdx);
  //  void fill_dedx_recoTrack_histos(reco::TrackRef trackref, std::vector< edm::ValueMap<reco::DeDxData> > v_dEdx);

  void fill_simAssociated_recoTrack_histos(int count,
					   const reco::Track& track);


  void fill_ResoAndPull_recoTrack_histos(int count,
					 ParticleBase::Vector momentumTP,
					 ParticleBase::Point vertexTP,
					 int chargeTP,
					 const reco::Track& track,
					 math::XYZPoint bsPosition);

  void finalHistoFits(int counter);


  void fillHistosFromVectors(int counter);
  void fillProfileHistosFromVectors(int counter);


 private:

  // private methods for internal usage
  void setUpVectors();


  /// retrieval of reconstructed momentum components from reco::Track (== mean values for GSF) 
  void getRecoMomentum (const reco::Track& track, double& pt, double& ptError,
			double& qoverp, double& qoverpError, double& lambda, double& lambdaError,  
			double& phi, double& phiError ) const;
  /// retrieval of reconstructed momentum components based on the mode of a reco::GsfTrack
  void getRecoMomentum (const reco::GsfTrack& gsfTrack, double& pt, double& ptError,
			double& qoverp, double& qoverpError, double& lambda, double& lambdaError,  
			double& phi, double& phiError) const;

  double getEta(double eta); 
  
  double getPt(double pt); 


  //private data members       
  TrackingParticleSelector* generalTpSelector;
  TrackingParticleSelector* TpSelectorForEfficiencyVsEta;
  TrackingParticleSelector* TpSelectorForEfficiencyVsPhi;
  TrackingParticleSelector* TpSelectorForEfficiencyVsPt;
  TrackingParticleSelector* TpSelectorForEfficiencyVsVTXR;
  TrackingParticleSelector* TpSelectorForEfficiencyVsVTXZ;

  double minEta, maxEta;  int nintEta;  bool useFabsEta;
  double minPt, maxPt;  int nintPt;   bool useInvPt;   bool useLogPt;
  double minHit, maxHit;  int nintHit;
  double minPhi, maxPhi;  int nintPhi;
  double minDxy, maxDxy;  int nintDxy;
  double minDz, maxDz;  int nintDz;
  double minVertpos, maxVertpos;  int nintVertpos;
  double minZpos, maxZpos;  int nintZpos;
  double minDeDx, maxDeDx;  int nintDeDx;

  //
  double ptRes_rangeMin,ptRes_rangeMax; int ptRes_nbin;
  double phiRes_rangeMin,phiRes_rangeMax; int phiRes_nbin;
  double cotThetaRes_rangeMin,cotThetaRes_rangeMax; int cotThetaRes_nbin;
  double dxyRes_rangeMin,dxyRes_rangeMax; int dxyRes_nbin;
  double dzRes_rangeMin,dzRes_rangeMax; int dzRes_nbin;


  //sim
  std::vector<MonitorElement*> h_ptSIM, h_etaSIM, h_tracksSIM, h_vertposSIM;
  
  //1D
  std::vector<MonitorElement*> h_tracks, h_fakes, h_hits, h_charge;
  std::vector<MonitorElement*> h_effic,  h_fakerate, h_recoeta, h_assoceta, h_assoc2eta, h_simuleta;
  std::vector<MonitorElement*> h_efficPt, h_fakeratePt, h_recopT, h_assocpT, h_assoc2pT, h_simulpT;
  std::vector<MonitorElement*> h_effic_vs_hit, h_fake_vs_hit, h_recohit, h_assochit, h_assoc2hit, h_simulhit;
  std::vector<MonitorElement*> h_effic_vs_phi, h_fake_vs_phi, h_recophi, h_assocphi, h_assoc2phi, h_simulphi;
  std::vector<MonitorElement*> h_effic_vs_dxy, h_fake_vs_dxy, h_recodxy, h_assocdxy, h_assoc2dxy, h_simuldxy;
  std::vector<MonitorElement*> h_effic_vs_dz, h_fake_vs_dz, h_recodz, h_assocdz, h_assoc2dz, h_simuldz;
  std::vector<MonitorElement*> h_effic_vs_vertpos, h_effic_vs_zpos, h_assocvertpos, h_simulvertpos, h_assoczpos, h_simulzpos;
  std::vector<MonitorElement*> h_pt, h_eta, h_pullTheta,h_pullPhi,h_pullDxy,h_pullDz,h_pullQoverp;

  // dE/dx
  // in the future these might become an array
  std::vector<MonitorElement*> h_dedx_estim1;
  std::vector<MonitorElement*> h_dedx_estim2;
  std::vector<MonitorElement*> h_dedx_nom1;
  std::vector<MonitorElement*> h_dedx_nom2;
  std::vector<MonitorElement*> h_dedx_sat1;
  std::vector<MonitorElement*> h_dedx_sat2;
  
  //2D  
  std::vector<MonitorElement*> nrec_vs_nsim;
  std::vector<MonitorElement*> nrecHit_vs_nsimHit_sim2rec;
  std::vector<MonitorElement*> nrecHit_vs_nsimHit_rec2sim;

  //assoc hits
  std::vector<MonitorElement*> h_assocFraction, h_assocSharedHit;
  
  //#hit vs eta: to be used with doProfileX
  std::vector<MonitorElement*> nhits_vs_eta, 
    nPXBhits_vs_eta, nPXFhits_vs_eta, 
    nTIBhits_vs_eta,nTIDhits_vs_eta,
    nTOBhits_vs_eta,nTEChits_vs_eta,
    nLayersWithMeas_vs_eta, nPXLlayersWithMeas_vs_eta, 
    nSTRIPlayersWithMeas_vs_eta, nSTRIPlayersWith1dMeas_vs_eta, nSTRIPlayersWith2dMeas_vs_eta;


  std::vector<MonitorElement*> h_hits_eta,
    h_PXBhits_eta, h_PXFhits_eta, h_TIBhits_eta,h_TIDhits_eta,
    h_TOBhits_eta,h_TEChits_eta,h_DThits_eta,h_CSChits_eta,h_RPChits_eta,
    h_LayersWithMeas_eta, h_PXLlayersWithMeas_eta, 
    h_STRIPlayersWithMeas_eta, h_STRIPlayersWith1dMeas_eta, h_STRIPlayersWith2dMeas_eta;
    

  std::vector< std::vector<double> > etaintervals;
  std::vector< std::vector<double> > pTintervals;
  std::vector< std::vector<double> > phiintervals;
  std::vector< std::vector<double> > dxyintervals;
  std::vector< std::vector<double> > dzintervals;
  std::vector< std::vector<double> > vertposintervals;
  std::vector< std::vector<double> > zposintervals;
  std::vector< std::vector<int> > totSIMeta,totRECeta,totASSeta,totASS2eta;
  std::vector< std::vector<int> > totSIMpT,totRECpT,totASSpT,totASS2pT;
  std::vector< std::vector<int> > totSIM_hit,totREC_hit,totASS_hit,totASS2_hit;
  std::vector< std::vector<int> > totSIM_phi,totREC_phi,totASS_phi,totASS2_phi;
  std::vector< std::vector<int> > totSIM_dxy,totREC_dxy,totASS_dxy,totASS2_dxy;
  std::vector< std::vector<int> > totSIM_dz,totREC_dz,totASS_dz,totASS2_dz;
  std::vector< std::vector<int> > totSIM_vertpos,totASS_vertpos,totSIM_zpos,totASS_zpos;
  


  //---- second set of histograms (originally not used by the SeedGenerator)
  //1D
  std::vector<MonitorElement*> h_nchi2, h_nchi2_prob, h_losthits;
  
  //2D  
  std::vector<MonitorElement*> chi2_vs_nhits, etares_vs_eta;
  std::vector<MonitorElement*> h_ptshifteta;
  std::vector<MonitorElement*> ptres_vs_phi, chi2_vs_phi, nhits_vs_phi, phires_vs_phi;

  //Profile2D
  std::vector<MonitorElement*> ptmean_vs_eta_phi, phimean_vs_eta_phi;
  
  //assoc chi2
  std::vector<MonitorElement*> h_assochi2, h_assochi2_prob;
  
  //chi2 and # lost hits vs eta: to be used with doProfileX
  std::vector<MonitorElement*> chi2_vs_eta, nlosthits_vs_eta;
  std::vector<MonitorElement*> h_chi2meanh, h_losthits_eta;
  std::vector<MonitorElement*> h_hits_phi;  
  std::vector<MonitorElement*> h_chi2meanhitsh, h_chi2mean_vs_phi;
  
  //resolution of track params: to be used with fitslicesytool
  std::vector<MonitorElement*> dxyres_vs_eta, ptres_vs_eta, dzres_vs_eta, phires_vs_eta, cotThetares_vs_eta;
  std::vector<MonitorElement*> dxyres_vs_pt, ptres_vs_pt, dzres_vs_pt, phires_vs_pt, cotThetares_vs_pt;

  std::vector<MonitorElement*> h_dxyrmsh, h_ptrmsh, h_dzrmsh, h_phirmsh, h_cotThetarmsh;
  std::vector<MonitorElement*> h_dxyrmshPt, h_ptrmshPt, h_dzrmshPt, h_phirmshPt, h_cotThetarmshPt;
  std::vector<MonitorElement*> h_ptrmshPhi, h_phirmshPhi;
  //  std::vector<MonitorElement*> h_phimeanh,h_ptmeanhhi, h_phimeanhPhi;
  
  std::vector<MonitorElement*> h_dxymeanh, h_ptmeanh, h_dzmeanh, h_phimeanh, h_cotThetameanh;
  std::vector<MonitorElement*> h_dxymeanhPt, h_ptmeanhPt, h_dzmeanhPt, h_phimeanhPt, h_cotThetameanhPt;
  std::vector<MonitorElement*> h_ptmeanhPhi, h_phimeanhPhi;

  //pulls of track params vs eta: to be used with fitslicesytool
  std::vector<MonitorElement*> dxypull_vs_eta, ptpull_vs_eta, dzpull_vs_eta, phipull_vs_eta, thetapull_vs_eta;
  std::vector<MonitorElement*> ptpull_vs_phi, phipull_vs_phi, thetapull_vs_phi;
  std::vector<MonitorElement*> h_dxypulleta, h_ptpulleta, h_dzpulleta, h_phipulleta, h_thetapulleta;
  std::vector<MonitorElement*> h_ptpullphi, h_phipullphi, h_thetapullphi;
  std::vector<MonitorElement*> h_dxypulletamean, h_ptpulletamean, h_dzpulletamean, h_phipulletamean, h_thetapulletamean;
  std::vector<MonitorElement*> h_ptpullphimean, h_phipullphimean, h_thetapullphimean;

};

#endif
