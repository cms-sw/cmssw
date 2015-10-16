#ifndef Validation_RecoTrack_MTVHistoProducerAlgoForTracker_h
#define Validation_RecoTrack_MTVHistoProducerAlgoForTracker_h

/* \author B.Mangano, UCSD
 *
 * Concrete class implementing the MTVHistoProducerAlgo interface.
 * To be used within the MTV to fill histograms for Tracker tracks.
 */

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/TrackReco/interface/DeDxData.h"
#include "DataFormats/Common/interface/ValueMap.h"

#include "SimTracker/Common/interface/TrackingParticleSelector.h"
#include "CommonTools/CandAlgos/interface/GenParticleCustomSelector.h"

#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"



class MTVHistoProducerAlgoForTracker {
 public:
  MTVHistoProducerAlgoForTracker(const edm::ParameterSet& pset, edm::ConsumesCollector && iC) :
    MTVHistoProducerAlgoForTracker(pset, iC) {}
  MTVHistoProducerAlgoForTracker(const edm::ParameterSet& pset, edm::ConsumesCollector & iC) ;
  ~MTVHistoProducerAlgoForTracker();

  void bookSimHistos(DQMStore::IBooker& ibook);
  void bookSimTrackHistos(DQMStore::IBooker& ibook);
  void bookSimTrackPVAssociationHistos(DQMStore::IBooker& ibook);
  void bookRecoHistos(DQMStore::IBooker& ibook);
  void bookRecoPVAssociationHistos(DQMStore::IBooker& ibook);
  void bookRecodEdxHistos(DQMStore::IBooker& ibook);


  void fill_generic_simTrack_histos(const TrackingParticle::Vector&,const TrackingParticle::Point& vertex, int bx);
  void fill_simTrackBased_histos(int numSimTracks);


  void fill_recoAssociated_simTrack_histos(int count,
					   const TrackingParticle& tp,
					   const TrackingParticle::Vector& momentumTP, const TrackingParticle::Point& vertexTP,
					   double dxy, double dz,
                                           double dxyPV, double dzPV,
                                           int nSimHits,
                                           int nSimLayers, int nSimPixelLayers, int nSimStripMonoAndStereoLayers,
					   const reco::Track* track,
					   int numVertices,
					   double dR,
					   const math::XYZPoint *pvPosition);

  void fill_recoAssociated_simTrack_histos(int count,
					   const reco::GenParticle& tp,
					   const TrackingParticle::Vector& momentumTP, const TrackingParticle::Point& vertexTP,
					   double dxy, double dz, int nSimHits,
					   const reco::Track* track,
					   int numVertices);


  void fill_generic_recoTrack_histos(int count,
				     const reco::Track& track,
				     const math::XYZPoint& bsPosition,
				     const math::XYZPoint *pvPosition,
				     bool isMatched,
				     bool isSigMatched,
				     bool isChargeMatched,
				     int numAssocRecoTracks,
				     int numVertices,
				     int nSimHits,
				     double sharedFraction,
				     double dR);

  void fill_dedx_recoTrack_histos(int count, const edm::RefToBase<reco::Track>& trackref, const std::vector< const edm::ValueMap<reco::DeDxData> *>& v_dEdx);

  void fill_simAssociated_recoTrack_histos(int count,
					   const reco::Track& track);

  void fill_trackBased_histos(int count,
		 	      int assTracks,
			      int numRecoTracks,
			      int numSimTracks);


  void fill_ResoAndPull_recoTrack_histos(int count,
					 const TrackingParticle::Vector& momentumTP,
					 const TrackingParticle::Point& vertexTP,
					 int chargeTP,
					 const reco::Track& track,
					 const math::XYZPoint& bsPosition);

 private:

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
  std::unique_ptr<TrackingParticleSelector> generalTpSelector;
  std::unique_ptr<TrackingParticleSelector> TpSelectorForEfficiencyVsEta;
  std::unique_ptr<TrackingParticleSelector> TpSelectorForEfficiencyVsPhi;
  std::unique_ptr<TrackingParticleSelector> TpSelectorForEfficiencyVsPt;
  std::unique_ptr<TrackingParticleSelector> TpSelectorForEfficiencyVsVTXR;
  std::unique_ptr<TrackingParticleSelector> TpSelectorForEfficiencyVsVTXZ;

  std::unique_ptr<GenParticleCustomSelector> generalGpSelector;
  std::unique_ptr<GenParticleCustomSelector> GpSelectorForEfficiencyVsEta;
  std::unique_ptr<GenParticleCustomSelector> GpSelectorForEfficiencyVsPhi;
  std::unique_ptr<GenParticleCustomSelector> GpSelectorForEfficiencyVsPt;
  std::unique_ptr<GenParticleCustomSelector> GpSelectorForEfficiencyVsVTXR;
  std::unique_ptr<GenParticleCustomSelector> GpSelectorForEfficiencyVsVTXZ;

  double minEta, maxEta;  int nintEta;  bool useFabsEta;
  double minPt, maxPt;  int nintPt;   bool useInvPt;   bool useLogPt;
  double minHit, maxHit;  int nintHit;
  double minPu, maxPu;  int nintPu;
  double minLayers, maxLayers;  int nintLayers;
  double minPhi, maxPhi;  int nintPhi;
  double minDxy, maxDxy;  int nintDxy;
  double minDz, maxDz;  int nintDz;
  double minVertpos, maxVertpos;  int nintVertpos;
  double minZpos, maxZpos;  int nintZpos;
  double mindr, maxdr;  int nintdr;
  double minChi2, maxChi2; int nintChi2;
  double minDeDx, maxDeDx;  int nintDeDx;
  double minVertcount, maxVertcount;  int nintVertcount;
  double minTracks, maxTracks; int nintTracks;

  //
  double ptRes_rangeMin,ptRes_rangeMax; int ptRes_nbin;
  double phiRes_rangeMin,phiRes_rangeMax; int phiRes_nbin;
  double cotThetaRes_rangeMin,cotThetaRes_rangeMax; int cotThetaRes_nbin;
  double dxyRes_rangeMin,dxyRes_rangeMax; int dxyRes_nbin;
  double dzRes_rangeMin,dzRes_rangeMax; int dzRes_nbin;

  double maxDzpvCum; int nintDzpvCum;
  double maxDzpvsigCum; int nintDzpvsigCum;

  //sim
  MonitorElement *h_ptSIM, *h_etaSIM, *h_tracksSIM, *h_vertposSIM, *h_bunchxSIM;

  //1D
  std::vector<MonitorElement*> h_tracks, h_fakes, h_hits, h_charge, h_algo;;
  std::vector<MonitorElement*> h_recoeta, h_assoceta, h_assoc2eta, h_simuleta, h_loopereta, h_misideta, h_pileupeta;
  std::vector<MonitorElement*> h_recopT, h_assocpT, h_assoc2pT, h_simulpT, h_looperpT, h_misidpT, h_pileuppT;
  std::vector<MonitorElement*> h_recohit, h_assochit, h_assoc2hit, h_simulhit, h_looperhit, h_misidhit, h_pileuphit;
  std::vector<MonitorElement*> h_recolayer, h_assoclayer, h_assoc2layer, h_simullayer, h_looperlayer, h_misidlayer, h_pileuplayer;
  std::vector<MonitorElement*> h_recopixellayer, h_assocpixellayer, h_assoc2pixellayer, h_simulpixellayer, h_looperpixellayer, h_misidpixellayer, h_pileuppixellayer;
  std::vector<MonitorElement*> h_reco3Dlayer, h_assoc3Dlayer, h_assoc23Dlayer, h_simul3Dlayer, h_looper3Dlayer, h_misid3Dlayer, h_pileup3Dlayer;
  std::vector<MonitorElement*> h_recopu, h_assocpu, h_assoc2pu, h_simulpu, h_looperpu, h_misidpu, h_pileuppu;
  std::vector<MonitorElement*> h_recophi, h_assocphi, h_assoc2phi, h_simulphi, h_looperphi, h_misidphi, h_pileupphi;
  std::vector<MonitorElement*> h_recodxy, h_assocdxy, h_assoc2dxy, h_simuldxy, h_looperdxy, h_misiddxy, h_pileupdxy;
  std::vector<MonitorElement*> h_recodz, h_assocdz, h_assoc2dz, h_simuldz, h_looperdz, h_misiddz, h_pileupdz;
  std::vector<MonitorElement*> h_recodxypv, h_assocdxypv, h_assoc2dxypv, h_simuldxypv, h_looperdxypv, h_misiddxypv, h_pileupdxypv;
  std::vector<MonitorElement*> h_recodzpv, h_assocdzpv, h_assoc2dzpv, h_simuldzpv, h_looperdzpv, h_misiddzpv, h_pileupdzpv;

  std::vector<MonitorElement*> h_assocvertpos, h_simulvertpos, h_assoczpos, h_simulzpos;
  std::vector<MonitorElement*> h_assocdr, h_assoc2dr, h_simuldr, h_recodr, h_pileupdr;
  std::vector<MonitorElement*> h_recochi2, h_assoc2chi2, h_looperchi2, h_misidchi2, h_pileupchi2;
  std::vector<MonitorElement*> h_pt, h_eta, h_pullTheta,h_pullPhi,h_pullDxy,h_pullDz,h_pullQoverp;
  std::vector<MonitorElement*> h_assoc2_itpu_eta, h_assoc2_itpu_sig_eta, h_assoc2eta_sig;
  std::vector<MonitorElement*> h_assoc2_itpu_vertcount, h_assoc2_itpu_sig_vertcount;
  std::vector<MonitorElement*> h_assoc2_ootpu_eta, h_assoc2_ootpu_vertcount;
  std::vector<MonitorElement*> h_reco_ootpu_eta, h_reco_ootpu_vertcount;
  std::vector<MonitorElement*> h_con_eta, h_con_vertcount, h_con_zpos;

  std::vector<MonitorElement*> h_reco_dzpvcut, h_assoc_dzpvcut, h_assoc2_dzpvcut, h_simul_dzpvcut, h_simul2_dzpvcut, h_pileup_dzpvcut;
  std::vector<MonitorElement*> h_reco_dzpvsigcut, h_assoc_dzpvsigcut, h_assoc2_dzpvsigcut, h_simul_dzpvsigcut, h_simul2_dzpvsigcut, h_pileup_dzpvsigcut;

  std::vector<MonitorElement*> h_reco_dzpvcut_pt, h_assoc_dzpvcut_pt, h_assoc2_dzpvcut_pt, h_simul_dzpvcut_pt, h_simul2_dzpvcut_pt, h_pileup_dzpvcut_pt;
  std::vector<MonitorElement*> h_reco_dzpvsigcut_pt, h_assoc_dzpvsigcut_pt, h_assoc2_dzpvsigcut_pt, h_simul_dzpvsigcut_pt, h_simul2_dzpvsigcut_pt, h_pileup_dzpvsigcut_pt;

  // dE/dx
  // in the future these might become an array
  std::vector<std::vector<MonitorElement*>> h_dedx_estim;
  std::vector<std::vector<MonitorElement*>> h_dedx_nom;
  std::vector<std::vector<MonitorElement*>> h_dedx_sat;

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


  //---- second set of histograms (originally not used by the SeedGenerator)
  //1D
  std::vector<MonitorElement*> h_nchi2, h_nchi2_prob, h_losthits, h_nmisslayers_inner, h_nmisslayers_outer;

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

  //resolution of track params: to be used with fitslicesytool
  std::vector<MonitorElement*> dxyres_vs_eta, ptres_vs_eta, dzres_vs_eta, phires_vs_eta, cotThetares_vs_eta;
  std::vector<MonitorElement*> dxyres_vs_pt, ptres_vs_pt, dzres_vs_pt, phires_vs_pt, cotThetares_vs_pt;

  //pulls of track params vs eta: to be used with fitslicesytool
  std::vector<MonitorElement*> dxypull_vs_eta, ptpull_vs_eta, dzpull_vs_eta, phipull_vs_eta, thetapull_vs_eta;
  std::vector<MonitorElement*> ptpull_vs_phi, phipull_vs_phi, thetapull_vs_phi;
};

#endif
