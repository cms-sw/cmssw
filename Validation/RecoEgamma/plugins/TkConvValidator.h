#ifndef TkConvValidator_H
#define TkConvValidator_H
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "RecoEgamma/EgammaMCTools/interface/PhotonMCTruthFinder.h"
#include "SimDataFormats/Associations/interface/TrackToTrackingParticleAssociator.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "DataFormats/Common/interface/RefToBaseVector.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/EgammaCandidates/interface/ConversionFwd.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"

//
//DQM services
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DQMOneEDAnalyzer.h"

//
#include <map>
#include <vector>

// forward declarations
class TFile;
class TH1F;
class TH2F;
class TProfile;
class TTree;
class SimVertex;
class SimTrack;
/** \class TkConvValidator
 **
 **
 **  $Id: TkConvValidator
 **  \author N.Marinelli - Univ. of Notre Dame
 **
 ***/

class TkConvValidator : public DQMOneEDAnalyzer<> {
public:
  //
  explicit TkConvValidator(const edm::ParameterSet&);
  ~TkConvValidator() override;

  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  void dqmBeginRun(edm::Run const& r, edm::EventSetup const& theEventSetup) override;
  void dqmEndRun(edm::Run const& r, edm::EventSetup const& es) override;
  void endJob() override;

private:
  //

  float phiNormalization(float& a);
  float etaTransformation(float a, float b);
  math::XYZVector recalculateMomentumAtFittedVertex(const MagneticField& mf,
                                                    const TrackerGeometry& trackerGeom,
                                                    const edm::RefToBase<reco::Track>& tk,
                                                    const reco::Vertex& vtx);

  std::string fName_;
  DQMStore* dbe_;
  edm::ESHandle<MagneticField> theMF_;

  int verbosity_;
  int nEvt_;
  int nEntry_;
  int nSimConv_[2];
  int nMatched_;
  int nRecConv_;
  int nRecConvAss_;
  int nRecConvAssWithEcal_;

  int nInvalidPCA_;

  edm::ParameterSet parameters_;
  edm::ESHandle<CaloGeometry> theCaloGeom_;
  edm::ESHandle<CaloTopology> theCaloTopo_;

  std::string conversionCollectionProducer_;
  std::string conversionCollection_;
  edm::EDGetTokenT<reco::ConversionCollection> conversionCollectionPr_Token_;

  std::string conversionTrackProducer_;

  std::string photonCollectionProducer_;
  std::string photonCollection_;
  edm::EDGetTokenT<reco::PhotonCollection> photonCollectionPr_Token_;

  edm::EDGetTokenT<reco::VertexCollection> offline_pvToken_;
  edm::EDGetTokenT<reco::BeamSpot> beamspotToken_;
  edm::EDGetTokenT<edm::SimTrackContainer> g4_simTk_Token_;
  edm::EDGetTokenT<edm::SimVertexContainer> g4_simVtx_Token_;
  edm::EDGetTokenT<TrackingParticleRefVector> tpSelForEff_Token_;
  edm::EDGetTokenT<TrackingParticleRefVector> tpSelForFake_Token_;
  edm::EDGetTokenT<edm::HepMCProduct> hepMC_Token_;
  edm::EDGetTokenT<reco::GenJetCollection> genjets_Token_;
  edm::EDGetTokenT<reco::TrackToTrackingParticleAssociator> trackAssociator_Token_;

  std::string dqmpath_;

  edm::InputTag label_tp_;

  PhotonMCTruthFinder* thePhotonMCTruthFinder_;

  bool isRunCentrally_;

  double minPhoEtCut_;
  double trkIsolExtRadius_;
  double trkIsolInnRadius_;
  double trkPtLow_;
  double lip_;
  double ecalIsolRadius_;
  double bcEtLow_;
  double hcalIsolExtRadius_;
  double hcalIsolInnRadius_;
  double hcalHitEtLow_;
  int numOfTracksInCone_;
  double trkPtSumCut_;
  double ecalEtSumCut_;
  double hcalEtSumCut_;
  bool dCotCutOn_;
  double dCotCutValue_;
  double dCotHardCutValue_;
  bool generalTracksOnly_;
  bool arbitratedMerged_;
  bool arbitratedEcalSeeded_;
  bool ecalalgotracks_;
  bool highPurity_;
  double minProb_;
  uint maxHitsBeforeVtx_;
  double minLxy_;

  /// global variable for the MC photon
  double mcPhi_;
  double mcEta_;
  double mcConvPt_;
  double mcConvR_;
  double mcConvZ_;
  double mcConvY_;
  double mcConvX_;
  double mcConvPhi_;
  double mcConvEta_;
  double mcJetEta_;
  double mcJetPhi_;

  edm::RefVector<TrackingParticleCollection> theConvTP_;
  //std::vector<TrackingParticleRef>    theConvTP_;

  double minPhoPtForEffic;
  double maxPhoEtaForEffic;
  double maxPhoZForEffic;
  double maxPhoRForEffic;
  double minPhoPtForPurity;
  double maxPhoEtaForPurity;
  double maxPhoZForPurity;
  double maxPhoRForPurity;

  double simMinPt_;
  double simMaxPt_;

  /// Global variables for reco Photon
  double recMinPt_;
  double recMaxPt_;

  //
  //
  MonitorElement* h_nSimConv_[2];
  MonitorElement* h_SimConvEtaPix_[2];
  //
  MonitorElement* h_simTkPt_;
  MonitorElement* h_simTkEta_;

  MonitorElement* h_simConvVtxRvsZ_[4];
  MonitorElement* h_simConvVtxYvsX_;

  ///   Denominator for efficiencies
  MonitorElement* h_AllSimConv_[5];
  MonitorElement* h_VisSimConv_[6];
  MonitorElement* h_VisSimConvLarge_;
  ///   Numerator for efficiencies
  MonitorElement* h_SimConvOneTracks_[5];
  MonitorElement* h_SimConvOneMTracks_[5];
  MonitorElement* h_SimConvTwoTracks_[5];
  MonitorElement* h_SimConvTwoMTracks_[5];
  MonitorElement* h_SimConvTwoMTracksAndVtxPGT0_[5];
  MonitorElement* h_SimConvTwoMTracksAndVtxPGT0005_[5];
  MonitorElement* h_SimConvTwoMTracksAndVtxPGT01_[5];
  // Numerator for double-counting quantification
  MonitorElement* h_SimRecConvOneTracks_[5];
  MonitorElement* h_SimRecConvOneMTracks_[5];
  MonitorElement* h_SimRecConvTwoTracks_[5];
  MonitorElement* h_SimRecConvTwoMTracks_[5];

  // Denominators for conversion fake rate
  MonitorElement* h_RecoConvTwoTracks_[5];
  // Numerators for conversion fake rate
  MonitorElement* h_RecoConvTwoMTracks_[5];

  /// info per conversion
  MonitorElement* h_nConv_[3][3];
  MonitorElement* h_convEtaMatchSC_[3][3];
  MonitorElement* h_convEta_[3][3];
  MonitorElement* h_convEta2_[3][3];
  MonitorElement* h_convPhi_[3][3];
  MonitorElement* h_convR_[3][3];
  MonitorElement* h_convZ_[3][3];
  MonitorElement* h_convPt_[3][3];
  MonitorElement* h_EoverPTracks_[3][3];

  MonitorElement* h_convSCdEta_[3][3];
  MonitorElement* h_convSCdPhi_[3][3];

  MonitorElement* h_convRplot_;
  MonitorElement* h_convZplot_;

  MonitorElement* h_convPtRes_[3];

  MonitorElement* h_invMass_[3][3];

  MonitorElement* h_DPhiTracksAtVtx_[3][3];
  MonitorElement* h2_DPhiTracksAtVtxVsEta_;
  MonitorElement* p_DPhiTracksAtVtxVsEta_;
  MonitorElement* h2_DPhiTracksAtVtxVsR_;
  MonitorElement* p_DPhiTracksAtVtxVsR_;

  MonitorElement* h_DCotTracks_[3][3];
  MonitorElement* h2_DCotTracksVsEta_;
  MonitorElement* p_DCotTracksVsEta_;
  MonitorElement* h2_DCotTracksVsR_;
  MonitorElement* p_DCotTracksVsR_;

  MonitorElement* h_distMinAppTracks_[3][3];

  MonitorElement* h_DPhiTracksAtEcal_[3][3];
  MonitorElement* h2_DPhiTracksAtEcalVsR_;
  MonitorElement* p_DPhiTracksAtEcalVsR_;
  MonitorElement* h2_DPhiTracksAtEcalVsEta_;
  MonitorElement* p_DPhiTracksAtEcalVsEta_;

  MonitorElement* h_DEtaTracksAtEcal_[3][3];

  MonitorElement* h_convVtxRvsZ_[3];
  MonitorElement* h_convVtxYvsX_;
  MonitorElement* h_convVtxRvsZ_zoom_[2];
  MonitorElement* h_convVtxYvsX_zoom_[2];

  MonitorElement* h_convVtxdX_;
  MonitorElement* h_convVtxdY_;
  MonitorElement* h_convVtxdZ_;
  MonitorElement* h_convVtxdR_;
  MonitorElement* h_convVtxdEta_;
  MonitorElement* h_convVtxdPhi_;

  MonitorElement* h_convVtxdX_barrel_;
  MonitorElement* h_convVtxdY_barrel_;
  MonitorElement* h_convVtxdZ_barrel_;
  MonitorElement* h_convVtxdR_barrel_;

  MonitorElement* h_convVtxdX_endcap_;
  MonitorElement* h_convVtxdY_endcap_;
  MonitorElement* h_convVtxdZ_endcap_;
  MonitorElement* h_convVtxdR_endcap_;

  MonitorElement* h2_convVtxdRVsR_;
  MonitorElement* h2_convVtxdRVsEta_;

  MonitorElement* p_convVtxdRVsR_;
  MonitorElement* p_convVtxdRVsEta_;

  MonitorElement* p_convVtxdXVsX_;
  MonitorElement* p_convVtxdYVsY_;
  MonitorElement* p_convVtxdZVsZ_;
  MonitorElement* p_convVtxdZVsR_;

  MonitorElement* p2_convVtxdRVsRZ_;
  MonitorElement* p2_convVtxdZVsRZ_;

  MonitorElement* h2_convVtxRrecVsTrue_;

  MonitorElement* h_vtxChi2Prob_[3][3];

  MonitorElement* h_zPVFromTracks_[2];
  MonitorElement* h_dzPVFromTracks_[2];
  MonitorElement* h2_dzPVVsR_;
  MonitorElement* p_dzPVVsR_;

  MonitorElement* h_lxybs_[3][3];
  MonitorElement* h_maxNHitsBeforeVtx_[3][3];
  MonitorElement* h_leadNHitsBeforeVtx_[3][3];
  MonitorElement* h_trailNHitsBeforeVtx_[3][3];
  MonitorElement* h_sumNHitsBeforeVtx_[3][3];
  MonitorElement* h_deltaExpectedHitsInner_[3][3];
  MonitorElement* h_leadExpectedHitsInner_[3][3];
  MonitorElement* h_maxDlClosestHitToVtx_[3][3];
  MonitorElement* h_maxDlClosestHitToVtxSig_[3][3];
  MonitorElement* h_nSharedHits_[3][3];

  //////////// info per track
  MonitorElement* nHits_[3];
  MonitorElement* p_nHitsVsEta_[3];
  MonitorElement* nHitsVsEta_[3];
  MonitorElement* p_nHitsVsR_[3];
  MonitorElement* nHitsVsR_[3];
  MonitorElement* h_tkChi2_[3];
  MonitorElement* h_tkChi2Large_[3];
  MonitorElement* h2_Chi2VsEta_[3];
  MonitorElement* p_Chi2VsEta_[3];
  MonitorElement* h2_Chi2VsR_[3];
  MonitorElement* p_Chi2VsR_[3];

  MonitorElement* h_TkD0_[3];

  MonitorElement* h_TkPtPull_[3];
  MonitorElement* h2_TkPtPull_[3];
  MonitorElement* p_TkPtPull_[3];
  MonitorElement* h2_PtRecVsPtSim_[3];
  MonitorElement* h2_photonPtRecVsPtSim_;

  MonitorElement* h_match_;

  MonitorElement* p2_effRZ_;

  MonitorElement* h_nHitsBeforeVtx_[3];
  MonitorElement* h_dlClosestHitToVtx_[3];
  MonitorElement* h_dlClosestHitToVtxSig_[3];
};

#endif
