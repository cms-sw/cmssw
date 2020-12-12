#ifndef Validation_RecoTrack_MTVHistoProducerAlgoForTracker_h
#define Validation_RecoTrack_MTVHistoProducerAlgoForTracker_h

/* \author B.Mangano, UCSD
 *
 * Concrete class implementing the MTVHistoProducerAlgo interface.
 * To be used within the MTV to fill histograms for Tracker tracks.
 */

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/TrackReco/interface/DeDxData.h"
#include "DataFormats/Common/interface/ValueMap.h"

#include "RecoTracker/TkSeedingLayers/interface/SeedingLayerSetsBuilder.h"

#include "SimTracker/Common/interface/TrackingParticleSelector.h"
#include "CommonTools/CandAlgos/interface/GenParticleCustomSelector.h"
#include "CommonTools/RecoAlgos/interface/RecoTrackSelectorBase.h"

#include "DQMServices/Core/interface/DQMStore.h"

struct MTVHistoProducerAlgoForTrackerHistograms {
  //sim
  using METype = dqm::reco::MonitorElement*;
  METype h_ptSIM, h_etaSIM, h_phiSIM, h_tracksSIM, h_vertposSIM, h_bunchxSIM;

  //1D
  std::vector<METype> h_tracks, h_fakes, h_hits, h_charge, h_algo, h_seedsFitFailed, h_seedsFitFailedFraction;
  std::vector<METype> h_recoeta, h_reco2eta, h_assoceta, h_assoc2eta, h_simuleta, h_loopereta, h_misideta, h_pileupeta;
  std::vector<METype> h_recopT, h_reco2pT, h_assocpT, h_assoc2pT, h_simulpT, h_looperpT, h_misidpT, h_pileuppT;
  std::vector<METype> h_recopTvseta, h_reco2pTvseta, h_assocpTvseta, h_assoc2pTvseta, h_simulpTvseta, h_looperpTvseta,
      h_misidpTvseta, h_pileuppTvseta;
  std::vector<METype> h_recohit, h_assochit, h_assoc2hit, h_simulhit, h_looperhit, h_misidhit, h_pileuphit;
  std::vector<METype> h_recolayer, h_assoclayer, h_assoc2layer, h_simullayer, h_looperlayer, h_misidlayer,
      h_pileuplayer;
  std::vector<METype> h_recopixellayer, h_assocpixellayer, h_assoc2pixellayer, h_simulpixellayer, h_looperpixellayer,
      h_misidpixellayer, h_pileuppixellayer;
  std::vector<METype> h_reco3Dlayer, h_assoc3Dlayer, h_assoc23Dlayer, h_simul3Dlayer, h_looper3Dlayer, h_misid3Dlayer,
      h_pileup3Dlayer;
  std::vector<METype> h_recopu, h_reco2pu, h_assocpu, h_assoc2pu, h_simulpu, h_looperpu, h_misidpu, h_pileuppu;
  std::vector<METype> h_recophi, h_assocphi, h_assoc2phi, h_simulphi, h_looperphi, h_misidphi, h_pileupphi;
  std::vector<METype> h_recodxy, h_assocdxy, h_assoc2dxy, h_simuldxy, h_looperdxy, h_misiddxy, h_pileupdxy;
  std::vector<METype> h_recodz, h_assocdz, h_assoc2dz, h_simuldz, h_looperdz, h_misiddz, h_pileupdz;
  std::vector<METype> h_recodxypv, h_assocdxypv, h_assoc2dxypv, h_simuldxypv, h_looperdxypv, h_misiddxypv,
      h_pileupdxypv;
  std::vector<METype> h_recodzpv, h_assocdzpv, h_assoc2dzpv, h_simuldzpv, h_looperdzpv, h_misiddzpv, h_pileupdzpv;
  std::vector<METype> h_recodxypvzoomed, h_assocdxypvzoomed, h_assoc2dxypvzoomed, h_simuldxypvzoomed,
      h_looperdxypvzoomed, h_misiddxypvzoomed, h_pileupdxypvzoomed;
  std::vector<METype> h_recodzpvzoomed, h_assocdzpvzoomed, h_assoc2dzpvzoomed, h_simuldzpvzoomed, h_looperdzpvzoomed,
      h_misiddzpvzoomed, h_pileupdzpvzoomed;

  std::vector<METype> h_recovertpos, h_assocvertpos, h_assoc2vertpos, h_simulvertpos, h_loopervertpos, h_pileupvertpos;
  std::vector<METype> h_recozpos, h_assoczpos, h_assoc2zpos, h_simulzpos, h_looperzpos, h_pileupzpos;
  std::vector<METype> h_assocdr, h_assoc2dr, h_simuldr, h_recodr, h_looperdr, h_pileupdr;
  std::vector<METype> h_assocdrj, h_assoc2drj, h_simuldrj, h_recodrj, h_looperdrj, h_pileupdrj;
  std::vector<METype> h_recochi2, h_assoc2chi2, h_looperchi2, h_misidchi2, h_pileupchi2;
  std::vector<METype> h_recochi2prob, h_assoc2chi2prob, h_looperchi2prob, h_misidchi2prob, h_pileupchi2prob;
  std::vector<METype> h_pt, h_eta, h_pullTheta, h_pullPhi, h_pullDxy, h_pullDz, h_pullQoverp;
  std::vector<METype> h_assoc2_itpu_eta, h_assoc2_itpu_sig_eta, h_assoc2eta_sig;
  std::vector<METype> h_assoc2_itpu_vertcount, h_assoc2_itpu_sig_vertcount;
  std::vector<METype> h_assoc2_ootpu_eta, h_assoc2_ootpu_vertcount;
  std::vector<METype> h_reco_ootpu_eta, h_reco_ootpu_vertcount;
  std::vector<METype> h_con_eta, h_con_vertcount, h_con_zpos;

  std::vector<METype> h_reco_dzpvcut, h_assoc_dzpvcut, h_assoc2_dzpvcut, h_simul_dzpvcut, h_simul2_dzpvcut,
      h_pileup_dzpvcut;
  std::vector<METype> h_reco_dzpvsigcut, h_assoc_dzpvsigcut, h_assoc2_dzpvsigcut, h_simul_dzpvsigcut,
      h_simul2_dzpvsigcut, h_pileup_dzpvsigcut;

  std::vector<METype> h_reco_simpvz, h_assoc_simpvz, h_assoc2_simpvz, h_simul_simpvz, h_looper_simpvz, h_pileup_simpvz;

  std::vector<METype> h_reco_seedingLayerSet, h_assoc2_seedingLayerSet, h_looper_seedingLayerSet,
      h_pileup_seedingLayerSet;

  std::vector<std::vector<METype>> h_reco_mva, h_assoc2_mva;
  std::vector<std::vector<METype>> h_reco_mvacut, h_assoc_mvacut, h_assoc2_mvacut, h_simul2_mvacut;
  std::vector<std::vector<METype>> h_reco_mva_hp, h_assoc2_mva_hp;
  std::vector<std::vector<METype>> h_reco_mvacut_hp, h_assoc_mvacut_hp, h_assoc2_mvacut_hp, h_simul2_mvacut_hp;

  std::vector<std::vector<METype>> h_assoc2_mva_vs_pt, h_fake_mva_vs_pt, h_assoc2_mva_vs_pt_hp, h_fake_mva_vs_pt_hp;
  std::vector<std::vector<METype>> h_assoc2_mva_vs_eta, h_fake_mva_vs_eta, h_assoc2_mva_vs_eta_hp, h_fake_mva_vs_eta_hp;

  // dE/dx
  // in the future these might become an array
  std::vector<std::vector<METype>> h_dedx_estim;
  std::vector<std::vector<METype>> h_dedx_nom;
  std::vector<std::vector<METype>> h_dedx_sat;

  //2D
  std::vector<METype> nrec_vs_nsim;
  std::vector<METype> nrecHit_vs_nsimHit_sim2rec;
  std::vector<METype> nrecHit_vs_nsimHit_rec2sim;
  std::vector<METype> h_duplicates_oriAlgo_vs_oriAlgo;

  //assoc hits
  std::vector<METype> h_assocFraction, h_assocSharedHit;

  //#hit vs eta: to be used with doProfileX
  std::vector<METype> nhits_vs_eta, nPXBhits_vs_eta, nPXFhits_vs_eta, nPXLhits_vs_eta, nTIBhits_vs_eta, nTIDhits_vs_eta,
      nTOBhits_vs_eta, nTEChits_vs_eta, nSTRIPhits_vs_eta, nLayersWithMeas_vs_eta, nPXLlayersWithMeas_vs_eta,
      nSTRIPlayersWithMeas_vs_eta, nSTRIPlayersWith1dMeas_vs_eta, nSTRIPlayersWith2dMeas_vs_eta, nMTDhits_vs_eta,
      nBTLhits_vs_eta, nETLhits_vs_eta;

  //---- second set of histograms (originally not used by the SeedGenerator)
  //1D
  std::vector<METype> h_nchi2, h_nchi2_prob, h_losthits, h_nmisslayers_inner, h_nmisslayers_outer;

  //2D
  std::vector<METype> chi2_vs_nhits, etares_vs_eta;
  std::vector<METype> h_ptshifteta;
  std::vector<METype> dxyres_vs_phi, dzres_vs_phi, ptres_vs_phi, chi2_vs_phi, nhits_vs_phi, phires_vs_phi;

  //Profile2D
  std::vector<METype> ptmean_vs_eta_phi, phimean_vs_eta_phi;

  //assoc chi2
  std::vector<METype> h_assochi2, h_assochi2_prob;

  //chi2 and # lost hits vs eta: to be used with doProfileX
  std::vector<METype> chi2_vs_eta, chi2_vs_pt, chi2_vs_drj, nlosthits_vs_eta;
  std::vector<METype> assoc_chi2_vs_eta, assoc_chi2_vs_pt, assoc_chi2_vs_drj, assoc_chi2prob_vs_eta,
      assoc_chi2prob_vs_pt, assoc_chi2prob_vs_drj;

  //resolution of track params: to be used with fitslicesytool
  std::vector<METype> dxyres_vs_eta, ptres_vs_eta, dzres_vs_eta, phires_vs_eta, cotThetares_vs_eta;
  std::vector<METype> dxyres_vs_pt, ptres_vs_pt, dzres_vs_pt, phires_vs_pt, cotThetares_vs_pt;

  //pulls of track params vs eta: to be used with fitslicesytool
  std::vector<METype> dxypull_vs_eta, ptpull_vs_eta, dzpull_vs_eta, phipull_vs_eta, thetapull_vs_eta;
  std::vector<METype> dxypull_vs_pt, ptpull_vs_pt, dzpull_vs_pt, phipull_vs_pt, thetapull_vs_pt;
  std::vector<METype> ptpull_vs_phi, phipull_vs_phi, thetapull_vs_phi;
};

class MTVHistoProducerAlgoForTracker {
public:
  typedef dqm::reco::DQMStore DQMStore;

  MTVHistoProducerAlgoForTracker(const edm::ParameterSet& pset, const bool doSeedPlots);
  ~MTVHistoProducerAlgoForTracker();

  static std::unique_ptr<RecoTrackSelectorBase> makeRecoTrackSelectorFromTPSelectorParameters(
      const edm::ParameterSet& pset);

  using Histograms = MTVHistoProducerAlgoForTrackerHistograms;
  void bookSimHistos(DQMStore::IBooker& ibook, Histograms& histograms);
  void bookSimTrackHistos(DQMStore::IBooker& ibook, Histograms& histograms, bool doResolutionPlots);
  void bookSimTrackPVAssociationHistos(DQMStore::IBooker& ibook, Histograms& histograms);
  void bookRecoHistos(DQMStore::IBooker& ibook, Histograms& histograms, bool doResolutionPlots);
  void bookRecoPVAssociationHistos(DQMStore::IBooker& ibook, Histograms& histograms);
  void bookRecodEdxHistos(DQMStore::IBooker& ibook, Histograms& histograms);
  void bookSeedHistos(DQMStore::IBooker& ibook, Histograms& histograms);
  void bookMVAHistos(DQMStore::IBooker& ibook, Histograms& histograms, size_t nMVAs);

  void fill_generic_simTrack_histos(const Histograms& histograms,
                                    const TrackingParticle::Vector&,
                                    const TrackingParticle::Point& vertex,
                                    int bx) const;
  void fill_simTrackBased_histos(const Histograms& histograms, int numSimTracks) const;

  void fill_recoAssociated_simTrack_histos(const Histograms& histograms,
                                           int count,
                                           const TrackingParticle& tp,
                                           const TrackingParticle::Vector& momentumTP,
                                           const TrackingParticle::Point& vertexTP,
                                           double dxy,
                                           double dz,
                                           double dxyPV,
                                           double dzPV,
                                           int nSimHits,
                                           int nSimLayers,
                                           int nSimPixelLayers,
                                           int nSimStripMonoAndStereoLayers,
                                           const reco::Track* track,
                                           int numVertices,
                                           double dR,
                                           double dR_jet,
                                           const math::XYZPoint* pvPosition,
                                           const TrackingVertex::LorentzVector* simPVPosition,
                                           const math::XYZPoint& bsPosition,
                                           const std::vector<float>& mvas,
                                           unsigned int selectsLoose,
                                           unsigned int selectsHP) const;

  void fill_recoAssociated_simTrack_histos(const Histograms& histograms,
                                           int count,
                                           const reco::GenParticle& tp,
                                           const TrackingParticle::Vector& momentumTP,
                                           const TrackingParticle::Point& vertexTP,
                                           double dxy,
                                           double dz,
                                           int nSimHits,
                                           const reco::Track* track,
                                           int numVertices) const;

  void fill_duplicate_histos(const Histograms& histograms,
                             int count,
                             const reco::Track& track1,
                             const reco::Track& track2) const;

  void fill_generic_recoTrack_histos(const Histograms& histograms,
                                     int count,
                                     const reco::Track& track,
                                     const TrackerTopology& ttopo,
                                     const math::XYZPoint& bsPosition,
                                     const math::XYZPoint* pvPosition,
                                     const TrackingVertex::LorentzVector* simPVPosition,
                                     bool isMatched,
                                     bool isSigMatched,
                                     bool isChargeMatched,
                                     int numAssocRecoTracks,
                                     int numVertices,
                                     int nSimHits,
                                     double sharedFraction,
                                     double dR,
                                     double dR_jet,
                                     const std::vector<float>& mvas,
                                     unsigned int selectsLoose,
                                     unsigned int selectsHP) const;

  void fill_dedx_recoTrack_histos(const Histograms& histograms,
                                  int count,
                                  const edm::RefToBase<reco::Track>& trackref,
                                  const std::vector<const edm::ValueMap<reco::DeDxData>*>& v_dEdx) const;

  void fill_simAssociated_recoTrack_histos(const Histograms& histograms, int count, const reco::Track& track) const;

  void fill_trackBased_histos(const Histograms& histograms,
                              int count,
                              int assTracks,
                              int numRecoTracks,
                              int numRecoTracksSelected,
                              int numSimTracksSelected) const;

  void fill_ResoAndPull_recoTrack_histos(const Histograms& histograms,
                                         int count,
                                         const TrackingParticle::Vector& momentumTP,
                                         const TrackingParticle::Point& vertexTP,
                                         int chargeTP,
                                         const reco::Track& track,
                                         const math::XYZPoint& bsPosition) const;

  void fill_seed_histos(const Histograms& histograms, int count, int seedsFitFailed, int seedsTotal) const;

private:
  /// retrieval of reconstructed momentum components from reco::Track (== mean values for GSF)
  void getRecoMomentum(const reco::Track& track,
                       double& pt,
                       double& ptError,
                       double& qoverp,
                       double& qoverpError,
                       double& lambda,
                       double& lambdaError,
                       double& phi,
                       double& phiError) const;
  /// retrieval of reconstructed momentum components based on the mode of a reco::GsfTrack
  void getRecoMomentum(const reco::GsfTrack& gsfTrack,
                       double& pt,
                       double& ptError,
                       double& qoverp,
                       double& qoverpError,
                       double& lambda,
                       double& lambdaError,
                       double& phi,
                       double& phiError) const;

  double getEta(double eta) const;

  double getPt(double pt) const;

  unsigned int getSeedingLayerSetBin(const reco::Track& track, const TrackerTopology& ttopo) const;

  //private data members
  std::unique_ptr<TrackingParticleSelector> generalTpSelector;
  std::unique_ptr<TrackingParticleSelector> TpSelectorForEfficiencyVsEta;
  std::unique_ptr<TrackingParticleSelector> TpSelectorForEfficiencyVsPhi;
  std::unique_ptr<TrackingParticleSelector> TpSelectorForEfficiencyVsPt;
  std::unique_ptr<TrackingParticleSelector> TpSelectorForEfficiencyVsVTXR;
  std::unique_ptr<TrackingParticleSelector> TpSelectorForEfficiencyVsVTXZ;

  std::unique_ptr<RecoTrackSelectorBase> trackSelectorVsEta;
  std::unique_ptr<RecoTrackSelectorBase> trackSelectorVsPhi;
  std::unique_ptr<RecoTrackSelectorBase> trackSelectorVsPt;

  std::unique_ptr<GenParticleCustomSelector> generalGpSelector;
  std::unique_ptr<GenParticleCustomSelector> GpSelectorForEfficiencyVsEta;
  std::unique_ptr<GenParticleCustomSelector> GpSelectorForEfficiencyVsPhi;
  std::unique_ptr<GenParticleCustomSelector> GpSelectorForEfficiencyVsPt;
  std::unique_ptr<GenParticleCustomSelector> GpSelectorForEfficiencyVsVTXR;
  std::unique_ptr<GenParticleCustomSelector> GpSelectorForEfficiencyVsVTXZ;

  double minEta, maxEta;
  int nintEta;
  bool useFabsEta;
  double minPt, maxPt;
  int nintPt;
  bool useInvPt;
  bool useLogPt;
  double minHit, maxHit;
  int nintHit;
  double minPu, maxPu;
  int nintPu;
  double minLayers, maxLayers;
  int nintLayers;
  double minPhi, maxPhi;
  int nintPhi;
  double minDxy, maxDxy;
  int nintDxy;
  double minDz, maxDz;
  int nintDz;
  double dxyDzZoom;
  double minVertpos, maxVertpos;
  int nintVertpos;
  bool useLogVertpos;
  double minZpos, maxZpos;
  int nintZpos;
  double mindr, maxdr;
  int nintdr;
  double mindrj, maxdrj;
  int nintdrj;
  double minChi2, maxChi2;
  int nintChi2;
  double minDeDx, maxDeDx;
  int nintDeDx;
  double minVertcount, maxVertcount;
  int nintVertcount;
  double minTracks, maxTracks;
  int nintTracks;
  double minPVz, maxPVz;
  int nintPVz;
  double minMVA, maxMVA;
  int nintMVA;

  const bool doSeedPlots_;
  const bool doMTDPlots_;

  //
  double ptRes_rangeMin, ptRes_rangeMax;
  int ptRes_nbin;
  double phiRes_rangeMin, phiRes_rangeMax;
  int phiRes_nbin;
  double cotThetaRes_rangeMin, cotThetaRes_rangeMax;
  int cotThetaRes_nbin;
  double dxyRes_rangeMin, dxyRes_rangeMax;
  int dxyRes_nbin;
  double dzRes_rangeMin, dzRes_rangeMax;
  int dzRes_nbin;

  double maxDzpvCum;
  int nintDzpvCum;
  double maxDzpvsigCum;
  int nintDzpvsigCum;

  std::vector<std::string> seedingLayerSetNames;
  using SeedingLayerId =
      std::tuple<SeedingLayerSetsBuilder::SeedingLayerId, bool>;  // last bool for strip mono (true) or not (false)
  using SeedingLayerSetId = std::array<SeedingLayerId, 4>;
  std::map<SeedingLayerSetId, unsigned int> seedingLayerSetToBin;
};

#endif
