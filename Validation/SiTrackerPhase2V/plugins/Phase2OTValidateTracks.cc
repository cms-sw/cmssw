
// Package:    Validation/SiTrackerPhase2V
// Class:      Phase2OTValidateTracks

/**
 * This class is part of the Phase 2 Tracker validation framework. It validates
 * the performance of L1 tracks reconstruction by comparing them with tracking
 * particles. It generates histograms to assess tracking efficiency, resolution,
 * and vertex reconstruction performance.
 *
 * Usage:
 * To generate histograms from this code, run the test configuration files
 * provided in the DQM/SiTrackerPhase2/test directory. The generated histograms
 * can then be analyzed or visualized.
 */

// Original Author:  Emily MacDonald

// Updated by: Brandi Skipworth, 2025

// system include files
#include <fstream>
#include <memory>
#include <numeric>
#include <vector>

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/L1TrackTrigger/interface/TTCluster.h"
#include "DataFormats/L1TrackTrigger/interface/TTStub.h"
#include "DataFormats/L1TrackTrigger/interface/TTTrack.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/CommonTopologies/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/Records/interface/StackedTrackerGeometryRecord.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "SimDataFormats/Associations/interface/TTClusterAssociationMap.h"
#include "SimDataFormats/Associations/interface/TTStubAssociationMap.h"
#include "SimDataFormats/Associations/interface/TTTrackAssociationMap.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "Validation/SiTrackerPhase2V/interface/TrackerPhase2ValidationUtil.h"

class Phase2OTValidateTracks : public DQMEDAnalyzer {
public:
  explicit Phase2OTValidateTracks(const edm::ParameterSet &);
  ~Phase2OTValidateTracks() override;
  void analyze(const edm::Event &, const edm::EventSetup &) override;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void dqmBeginRun(const edm::Run &iRun, const edm::EventSetup &iSetup) override;
  void processTrackCollection(const edm::Handle<TTTrackAssociationMap<Ref_Phase2TrackerDigi_>> &trackHandle,
                              edm::Ptr<TrackingParticle> tp_ptr,
                              bool isExtended,
                              int tmp_tp_pdgid,
                              float tmp_tp_d0,
                              float tmp_tp_Lxy,
                              float tmp_tp_pt,
                              float tmp_tp_eta,
                              float tmp_tp_phi,
                              float tmp_tp_z0,
                              const edm::Event &iEvent);
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);
  // Tracking particle distributions
  MonitorElement *trackParts_Eta = nullptr;
  MonitorElement *trackParts_Phi = nullptr;
  MonitorElement *trackParts_Pt = nullptr;
  MonitorElement *n_trackParts = nullptr;
  unsigned int nTrackParts = 0;

  // pT and eta for efficiency plots
  MonitorElement *tp_pt = nullptr;             // denominator
  MonitorElement *tp_pt_zoom = nullptr;        // denominator
  MonitorElement *tp_eta = nullptr;            // denominator
  MonitorElement *tp_d0 = nullptr;             // denominator
  MonitorElement *tp_Lxy = nullptr;            // denominator (also known as vxy)
  MonitorElement *tp_z0 = nullptr;             // denominator
  MonitorElement *match_tp_pt = nullptr;       // numerator
  MonitorElement *match_tp_pt_zoom = nullptr;  // numerator
  MonitorElement *match_tp_eta = nullptr;      // numerator
  MonitorElement *match_tp_d0 = nullptr;       // numerator
  MonitorElement *match_tp_Lxy = nullptr;      // numerator (also known as vxy)
  MonitorElement *match_tp_z0 = nullptr;       // numerator

  // extended displaced plots for denom with no Lxy cut
  MonitorElement *tp_pt_for_dis = nullptr;       // denominator
  MonitorElement *tp_pt_zoom_for_dis = nullptr;  // denominator
  MonitorElement *tp_eta_for_dis = nullptr;      // denominator
  MonitorElement *tp_d0_for_dis = nullptr;       // denominator
  MonitorElement *tp_z0_for_dis = nullptr;       // denominator

  // extended tracks pT and eta for efficiency plots
  MonitorElement *match_prompt_tp_pt = nullptr;          // numerator
  MonitorElement *match_prompt_tp_pt_zoom = nullptr;     // numerator
  MonitorElement *match_prompt_tp_eta = nullptr;         // numerator
  MonitorElement *match_prompt_tp_d0 = nullptr;          // numerator
  MonitorElement *match_prompt_tp_Lxy = nullptr;         // numerator (also known as vxy)
  MonitorElement *match_prompt_tp_z0 = nullptr;          // numerator
  MonitorElement *match_displaced_tp_pt = nullptr;       // numerator
  MonitorElement *match_displaced_tp_pt_zoom = nullptr;  // numerator
  MonitorElement *match_displaced_tp_eta = nullptr;      // numerator
  MonitorElement *match_displaced_tp_d0 = nullptr;       // numerator
  MonitorElement *match_displaced_tp_Lxy = nullptr;      // numerator (also known as vxy)
  MonitorElement *match_displaced_tp_z0 = nullptr;       // numerator

  MonitorElement *d0_res_hist = nullptr;
  MonitorElement *res_displaced_d0 = nullptr;
  MonitorElement *res_prompt_d0 = nullptr;

  // 1D intermediate resolution plots (pT and eta)
  MonitorElement *res_eta = nullptr;    // for all eta and pT
  MonitorElement *res_pt = nullptr;     // for all eta and pT
  MonitorElement *res_ptRel = nullptr;  // for all eta and pT (delta(pT)/pT)

  // 1D intermediate resolution plots for extended (pT and eta)
  // prompt tracks d0 < 0.1
  MonitorElement *res_prompt_eta = nullptr;
  MonitorElement *res_prompt_pt = nullptr;
  MonitorElement *res_prompt_ptRel = nullptr;
  // displaced tracks d0 >= 0.1
  MonitorElement *res_displaced_eta = nullptr;    // for all eta and pT
  MonitorElement *res_displaced_pt = nullptr;     // for all eta and pT
  MonitorElement *res_displaced_ptRel = nullptr;  // for all eta and pT (delta(pT)/pT)

  // Regular L1 track histograms
  std::vector<MonitorElement *> respt_pt2to3;
  std::vector<MonitorElement *> respt_pt3to8;
  std::vector<MonitorElement *> respt_pt8toInf;
  std::vector<MonitorElement *> reseta_vect;
  std::vector<MonitorElement *> resphi_vect;
  std::vector<MonitorElement *> resz0_vect;
  std::vector<MonitorElement *> resd0_vect;

  // Extended track histograms
  std::vector<MonitorElement *> reseta_prompt_vect;
  std::vector<MonitorElement *> resphi_prompt_vect;
  std::vector<MonitorElement *> resz0_prompt_vect;
  std::vector<MonitorElement *> resd0_prompt_vect;
  std::vector<MonitorElement *> respt_prompt_pt2to3;
  std::vector<MonitorElement *> respt_prompt_pt3to8;
  std::vector<MonitorElement *> respt_prompt_pt8toInf;
  std::vector<MonitorElement *> respt_displaced_pt2to3;
  std::vector<MonitorElement *> respt_displaced_pt3to8;
  std::vector<MonitorElement *> respt_displaced_pt8toInf;
  std::vector<MonitorElement *> reseta_displaced_vect;
  std::vector<MonitorElement *> resphi_displaced_vect;
  std::vector<MonitorElement *> resz0_displaced_vect;
  std::vector<MonitorElement *> resd0_displaced_vect;

private:
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> m_topoToken;
  edm::ParameterSet conf_;
  edm::EDGetTokenT<std::vector<TrackingParticle>> trackingParticleToken_;
  edm::EDGetTokenT<TTClusterAssociationMap<Ref_Phase2TrackerDigi_>>
      ttClusterMCTruthToken_;  // MC truth association map for clusters
  edm::EDGetTokenT<TTStubAssociationMap<Ref_Phase2TrackerDigi_>>
      ttStubMCTruthToken_;  // MC truth association map for stubs
  edm::EDGetTokenT<TTTrackAssociationMap<Ref_Phase2TrackerDigi_>>
      ttTrackMCTruthToken_;  // MC truth association map for tracks
  edm::EDGetTokenT<TTTrackAssociationMap<Ref_Phase2TrackerDigi_>>
      ttTrackMCTruthExtendedToken_;  // MC truth association map for extended
                                     // tracks
  edm::EDGetTokenT<edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_>>> ttStubToken_;  // L1 Stub token
  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> getTokenTrackerGeom_;     // Tracker geometry token
  int L1Tk_minNStub;
  double L1Tk_maxChi2dof;
  int TP_minNStub;
  int TP_minNLayersStub;
  double TP_minPt;
  double TP_maxEta;
  double TP_maxZ0;
  double TP_maxLxy;
  double TP_maxD0;
  std::string topFolderName_;
};

//
// constructors and destructor
//
Phase2OTValidateTracks::Phase2OTValidateTracks(const edm::ParameterSet &iConfig)
    : m_topoToken(esConsumes()), conf_(iConfig) {
  topFolderName_ = conf_.getParameter<std::string>("TopFolderName");
  trackingParticleToken_ =
      consumes<std::vector<TrackingParticle>>(conf_.getParameter<edm::InputTag>("trackingParticleToken"));
  ttStubMCTruthToken_ =
      consumes<TTStubAssociationMap<Ref_Phase2TrackerDigi_>>(conf_.getParameter<edm::InputTag>("MCTruthStubInputTag"));
  ttClusterMCTruthToken_ = consumes<TTClusterAssociationMap<Ref_Phase2TrackerDigi_>>(
      conf_.getParameter<edm::InputTag>("MCTruthClusterInputTag"));
  ttTrackMCTruthToken_ = consumes<TTTrackAssociationMap<Ref_Phase2TrackerDigi_>>(
      conf_.getParameter<edm::InputTag>("MCTruthTrackInputTag"));
  ttTrackMCTruthExtendedToken_ = consumes<TTTrackAssociationMap<Ref_Phase2TrackerDigi_>>(
      conf_.getParameter<edm::InputTag>("MCTruthTrackExtendedInputTag"));
  ttStubToken_ = consumes<edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_>>>(
      edm::InputTag("TTStubsFromPhase2TrackerDigis", "StubAccepted"));
  getTokenTrackerGeom_ = esConsumes<TrackerGeometry, TrackerDigiGeometryRecord>();
  L1Tk_minNStub = conf_.getParameter<int>("L1Tk_minNStub");         // min number of stubs in the track
  L1Tk_maxChi2dof = conf_.getParameter<double>("L1Tk_maxChi2dof");  // maximum chi2/dof of the track
  TP_minNStub = conf_.getParameter<int>("TP_minNStub");  // Minimum number of stubs associated to the tracking
                                                         // particle
  TP_minNLayersStub = conf_.getParameter<int>("TP_minNLayersStub");  // Minimum number of layers with stubs for the
                                                                     // tracking particle to be considered for matching
  TP_minNLayersStub = conf_.getParameter<int>("TP_minNLayersStub");
  TP_minPt = conf_.getParameter<double>("TP_minPt");    // min pT to consider matching
  TP_maxEta = conf_.getParameter<double>("TP_maxEta");  // max eta to consider matching
  TP_maxZ0 = conf_.getParameter<double>("TP_maxZ0");    // max vertZ (or z0) to consider matching
  TP_maxLxy = conf_.getParameter<double>("TP_maxLxy");  // max Lxy to consider prompt matching
  TP_maxD0 = conf_.getParameter<double>("TP_maxD0");    // max d0 to consider prompt matching
}

Phase2OTValidateTracks::~Phase2OTValidateTracks() = default;

void Phase2OTValidateTracks::dqmBeginRun(const edm::Run &iRun, const edm::EventSetup &iSetup) {
  // Clear existing histograms
  respt_pt2to3.clear();
  respt_pt3to8.clear();
  respt_pt8toInf.clear();
  reseta_vect.clear();
  resphi_vect.clear();
  resz0_vect.clear();
  resd0_vect.clear();

  respt_prompt_pt2to3.clear();
  respt_prompt_pt3to8.clear();
  respt_prompt_pt8toInf.clear();
  reseta_prompt_vect.clear();
  resphi_prompt_vect.clear();
  resz0_prompt_vect.clear();
  resd0_prompt_vect.clear();

  respt_displaced_pt2to3.clear();
  respt_displaced_pt3to8.clear();
  respt_displaced_pt8toInf.clear();
  reseta_displaced_vect.clear();
  resphi_displaced_vect.clear();
  resz0_displaced_vect.clear();
  resd0_displaced_vect.clear();

  // Resize vectors and set elements to nullptr
  respt_pt2to3.resize(6, nullptr);
  respt_pt3to8.resize(6, nullptr);
  respt_pt8toInf.resize(6, nullptr);
  reseta_vect.resize(6, nullptr);
  resphi_vect.resize(6, nullptr);
  resz0_vect.resize(6, nullptr);
  resd0_vect.resize(6, nullptr);

  respt_prompt_pt2to3.resize(6, nullptr);
  respt_prompt_pt3to8.resize(6, nullptr);
  respt_prompt_pt8toInf.resize(6, nullptr);
  reseta_prompt_vect.resize(6, nullptr);
  resphi_prompt_vect.resize(6, nullptr);
  resz0_prompt_vect.resize(6, nullptr);
  resd0_prompt_vect.resize(6, nullptr);

  respt_displaced_pt2to3.resize(6, nullptr);
  respt_displaced_pt3to8.resize(6, nullptr);
  respt_displaced_pt8toInf.resize(6, nullptr);
  reseta_displaced_vect.resize(6, nullptr);
  resphi_displaced_vect.resize(6, nullptr);
  resz0_displaced_vect.resize(6, nullptr);
  resd0_displaced_vect.resize(6, nullptr);
}
// member functions

// ---------------------------------------------------------------------
// Helper function to process a track collection (nominal or extended)
// ---------------------------------------------------------------------
void Phase2OTValidateTracks::processTrackCollection(
    const edm::Handle<TTTrackAssociationMap<Ref_Phase2TrackerDigi_>> &trackHandle,
    edm::Ptr<TrackingParticle> tp_ptr,
    bool isExtended,
    int tmp_tp_pdgid,
    float tmp_tp_d0,
    float tmp_tp_Lxy,
    float tmp_tp_pt,
    float tmp_tp_eta,
    float tmp_tp_phi,
    float tmp_tp_z0,
    const edm::Event &iEvent) {
  std::vector<MonitorElement *> *reseta_vect_ptr = nullptr;
  std::vector<MonitorElement *> *resphi_vect_ptr = nullptr;
  std::vector<MonitorElement *> *resz0_vect_ptr = nullptr;
  std::vector<MonitorElement *> *resd0_vect_ptr = nullptr;
  std::vector<MonitorElement *> *respt_pt2to3_ptr = nullptr;
  std::vector<MonitorElement *> *respt_pt3to8_ptr = nullptr;
  std::vector<MonitorElement *> *respt_pt8toInf_ptr = nullptr;

  MonitorElement *res_eta_ptr = nullptr;
  MonitorElement *res_pt_ptr = nullptr;
  MonitorElement *res_ptRel_ptr = nullptr;
  MonitorElement *match_tp_pt_ptr = nullptr;
  MonitorElement *match_tp_pt_zoom_ptr = nullptr;
  MonitorElement *match_tp_eta_ptr = nullptr;
  MonitorElement *match_tp_d0_ptr = nullptr;
  MonitorElement *match_tp_Lxy_ptr = nullptr;
  MonitorElement *match_tp_z0_ptr = nullptr;

  MonitorElement *res_d0_hist_ptr = nullptr;

  std::vector<edm::Ptr<TTTrack<Ref_Phase2TrackerDigi_>>> matchedTracks = trackHandle->findTTTrackPtrs(tp_ptr);

  int tp_nMatch = 0;
  int bestTrackIndex = -1;
  float bestChi2dof = std::numeric_limits<float>::max();

  for (size_t i = 0; i < matchedTracks.size(); i++) {
    const auto &thisTrack = matchedTracks[i];

    if (!trackHandle->isGenuine(thisTrack))
      continue;

    int nStubs = thisTrack->getStubRefs().size();
    if (nStubs < L1Tk_minNStub)
      continue;

    auto associatedTP = trackHandle->findTrackingParticlePtr(thisTrack);
    if (!associatedTP.isNonnull())
      continue;

    float dmatch_pt = std::fabs(associatedTP->p4().pt() - tmp_tp_pt);
    float dmatch_eta = std::fabs(associatedTP->p4().eta() - tmp_tp_eta);
    float dmatch_phi = std::fabs(associatedTP->p4().phi() - tmp_tp_phi);
    int match_id = associatedTP->pdgId();
    float chi2dof = thisTrack->chi2Red();

    if (dmatch_pt < 0.1 && dmatch_eta < 0.1 && dmatch_phi < 0.1 && tmp_tp_pdgid == match_id) {
      tp_nMatch++;
      if (bestTrackIndex < 0 || chi2dof < bestChi2dof) {
        bestTrackIndex = i;
        bestChi2dof = chi2dof;
      }
    }
  }  // end loop over matched tracks

  if (tp_nMatch < 1)
    return;

  const auto &bestTrack = matchedTracks[bestTrackIndex];

  float bestTrack_d0 = -bestTrack->POCA().x() * sin(bestTrack->momentum().phi()) +
                       bestTrack->POCA().y() * cos(bestTrack->momentum().phi());

  int nStubs = bestTrack->getStubRefs().size();
  float chi2dof = bestTrack->chi2Red();
  if (nStubs < L1Tk_minNStub || chi2dof > L1Tk_maxChi2dof)
    return;

  if (isExtended) {
    // if extended, if "prompt" or "displaced"
    if (std::fabs(bestTrack_d0) < TP_maxD0 && tmp_tp_Lxy < TP_maxLxy) {
      // extended + prompt
      reseta_vect_ptr = &reseta_prompt_vect;
      resphi_vect_ptr = &resphi_prompt_vect;
      resz0_vect_ptr = &resz0_prompt_vect;
      resd0_vect_ptr = &resd0_prompt_vect;
      respt_pt2to3_ptr = &respt_prompt_pt2to3;
      respt_pt3to8_ptr = &respt_prompt_pt3to8;
      respt_pt8toInf_ptr = &respt_prompt_pt8toInf;

      res_eta_ptr = res_prompt_eta;
      res_pt_ptr = res_prompt_pt;
      res_ptRel_ptr = res_prompt_ptRel;
      res_d0_hist_ptr = res_prompt_d0;

      match_tp_pt_ptr = match_prompt_tp_pt;
      match_tp_pt_zoom_ptr = match_prompt_tp_pt_zoom;
      match_tp_eta_ptr = match_prompt_tp_eta;
      match_tp_d0_ptr = match_prompt_tp_d0;
      match_tp_Lxy_ptr = match_prompt_tp_Lxy;
      match_tp_z0_ptr = match_prompt_tp_z0;
    } else {
      // extended + displaced
      reseta_vect_ptr = &reseta_displaced_vect;
      resphi_vect_ptr = &resphi_displaced_vect;
      resz0_vect_ptr = &resz0_displaced_vect;
      resd0_vect_ptr = &resd0_displaced_vect;
      respt_pt2to3_ptr = &respt_displaced_pt2to3;
      respt_pt3to8_ptr = &respt_displaced_pt3to8;
      respt_pt8toInf_ptr = &respt_displaced_pt8toInf;

      res_eta_ptr = res_displaced_eta;
      res_pt_ptr = res_displaced_pt;
      res_ptRel_ptr = res_displaced_ptRel;
      res_d0_hist_ptr = res_displaced_d0;

      match_tp_pt_ptr = match_displaced_tp_pt;
      match_tp_pt_zoom_ptr = match_displaced_tp_pt_zoom;
      match_tp_eta_ptr = match_displaced_tp_eta;
      match_tp_d0_ptr = match_displaced_tp_d0;
      match_tp_Lxy_ptr = match_displaced_tp_Lxy;
      match_tp_z0_ptr = match_displaced_tp_z0;
    }

  } else {
    // Nominal (non‐extended)
    reseta_vect_ptr = &reseta_vect;
    resphi_vect_ptr = &resphi_vect;
    resz0_vect_ptr = &resz0_vect;
    resd0_vect_ptr = &resd0_vect;
    respt_pt2to3_ptr = &respt_pt2to3;
    respt_pt3to8_ptr = &respt_pt3to8;
    respt_pt8toInf_ptr = &respt_pt8toInf;

    res_eta_ptr = res_eta;
    res_pt_ptr = res_pt;
    res_ptRel_ptr = res_ptRel;

    match_tp_pt_ptr = match_tp_pt;
    match_tp_pt_zoom_ptr = match_tp_pt_zoom;
    match_tp_eta_ptr = match_tp_eta;
    match_tp_d0_ptr = match_tp_d0;
    match_tp_Lxy_ptr = match_tp_Lxy;
    match_tp_z0_ptr = match_tp_z0;

    res_d0_hist_ptr = d0_res_hist;
  }

  // Compute residuals & fill them
  float eta_res = bestTrack->momentum().eta() - tmp_tp_eta;
  float phi_res = bestTrack->momentum().phi() - tmp_tp_phi;
  float z0_res = bestTrack->z0() - tmp_tp_z0;
  float d0_res = bestTrack_d0 - tmp_tp_d0;

  float pt_diff = bestTrack->momentum().perp() - tmp_tp_pt;
  float pt_res = (tmp_tp_pt != 0.f) ? (pt_diff / tmp_tp_pt) : 0.f;

  res_eta_ptr->Fill(eta_res);
  res_pt_ptr->Fill(pt_diff);
  res_ptRel_ptr->Fill(pt_res);

  res_d0_hist_ptr->Fill(d0_res);

  // Fill efficiency histograms
  match_tp_pt_ptr->Fill(tmp_tp_pt);
  if (tmp_tp_pt > 0.f && tmp_tp_pt <= 10.f)
    match_tp_pt_zoom_ptr->Fill(tmp_tp_pt);
  match_tp_eta_ptr->Fill(tmp_tp_eta);
  match_tp_d0_ptr->Fill(tmp_tp_d0);
  match_tp_Lxy_ptr->Fill(tmp_tp_Lxy);
  match_tp_z0_ptr->Fill(tmp_tp_z0);

  // Fill resolution plots for different abs(eta) bins:
  float bins[7] = {0.f, 0.7f, 1.0f, 1.2f, 1.6f, 2.0f, 2.4f};
  for (int i = 0; i < 6; i++) {
    if (std::fabs(tmp_tp_eta) >= bins[i] && std::fabs(tmp_tp_eta) < bins[i + 1]) {
      // Use pointer dereferences to fill
      (*reseta_vect_ptr)[i]->Fill(eta_res);
      (*resphi_vect_ptr)[i]->Fill(phi_res);
      (*resz0_vect_ptr)[i]->Fill(z0_res);
      (*resd0_vect_ptr)[i]->Fill(d0_res);

      if (tmp_tp_pt >= 2.f && tmp_tp_pt < 3.f) {
        (*respt_pt2to3_ptr)[i]->Fill(pt_res);
      } else if (tmp_tp_pt >= 3.f && tmp_tp_pt < 8.f) {
        (*respt_pt3to8_ptr)[i]->Fill(pt_res);
      } else if (tmp_tp_pt >= 8.f) {
        (*respt_pt8toInf_ptr)[i]->Fill(pt_res);
      }
      break;
    }
  }
}

// ------------ method called for each event  ------------
void Phase2OTValidateTracks::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  // Tracking Particles
  edm::Handle<std::vector<TrackingParticle>> trackingParticleHandle;
  iEvent.getByToken(trackingParticleToken_, trackingParticleHandle);

  // Truth Association Maps
  edm::Handle<TTTrackAssociationMap<Ref_Phase2TrackerDigi_>> MCTruthTTTrackHandle;
  iEvent.getByToken(ttTrackMCTruthToken_, MCTruthTTTrackHandle);
  edm::Handle<TTTrackAssociationMap<Ref_Phase2TrackerDigi_>> MCTruthTTTrackExtendedHandle;
  iEvent.getByToken(ttTrackMCTruthExtendedToken_, MCTruthTTTrackExtendedHandle);
  edm::Handle<TTClusterAssociationMap<Ref_Phase2TrackerDigi_>> MCTruthTTClusterHandle;
  iEvent.getByToken(ttClusterMCTruthToken_, MCTruthTTClusterHandle);
  edm::Handle<TTStubAssociationMap<Ref_Phase2TrackerDigi_>> MCTruthTTStubHandle;
  iEvent.getByToken(ttStubMCTruthToken_, MCTruthTTStubHandle);
  edm::Handle<edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_>>> TTStubHandle;
  iEvent.getByToken(ttStubToken_, TTStubHandle);

  // Geometries
  const TrackerTopology *const tTopo = &iSetup.getData(m_topoToken);

  // Loop over tracking particles
  int this_tp = 0;
  for (const auto &iterTP : *trackingParticleHandle) {
    edm::Ptr<TrackingParticle> tp_ptr(trackingParticleHandle, this_tp);
    this_tp++;

    // int tmp_eventid = iterTP.eventId().event();
    float tmp_tp_pt = iterTP.pt();
    float tmp_tp_phi = iterTP.phi();
    float tmp_tp_eta = iterTP.eta();

    // Calculate nLayers variable
    std::vector<edm::Ref<edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_>>, TTStub<Ref_Phase2TrackerDigi_>>>
        theStubRefs = MCTruthTTStubHandle->findTTStubRefs(tp_ptr);

    int hasStubInLayer[11] = {0};
    for (unsigned int is = 0; is < theStubRefs.size(); is++) {
      DetId detid(theStubRefs.at(is)->getDetId());
      int layer = -1;
      if (detid.subdetId() == StripSubdetector::TOB)
        layer = static_cast<int>(tTopo->layer(detid)) - 1;  // fill in array as entries 0-5
      else if (detid.subdetId() == StripSubdetector::TID)
        layer = static_cast<int>(tTopo->layer(detid)) + 5;  // fill in array as entries 6-10

      // treat genuine stubs separately (==2 is genuine, ==1 is not)
      if (MCTruthTTStubHandle->findTrackingParticlePtr(theStubRefs.at(is)).isNull() && hasStubInLayer[layer] < 2)
        hasStubInLayer[layer] = 1;
      else
        hasStubInLayer[layer] = 2;
    }

    int nStubLayerTP = 0;
    for (int isum = 0; isum < 11; isum++) {
      if (hasStubInLayer[isum] >= 1)
        nStubLayerTP += 1;
    }

    if (std::fabs(tmp_tp_eta) > TP_maxEta)
      continue;
    // Fill the 1D distribution plots for tracking particles, to monitor change
    // in stub definition
    if (tmp_tp_pt > TP_minPt && nStubLayerTP >= TP_minNLayersStub) {
      trackParts_Pt->Fill(tmp_tp_pt);
      trackParts_Eta->Fill(tmp_tp_eta);
      trackParts_Phi->Fill(tmp_tp_phi);
      nTrackParts++;
    }

    // if (TP_select_eventid == 0 && tmp_eventid != 0)
    //   continue;  //only care about tracking particles from the primary
    //   interaction for efficiency/resolution
    int nStubTP = -1;
    if (MCTruthTTStubHandle.isValid()) {
      std::vector<edm::Ref<edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_>>, TTStub<Ref_Phase2TrackerDigi_>>>
          theStubRefs = MCTruthTTStubHandle->findTTStubRefs(tp_ptr);
      nStubTP = (int)theStubRefs.size();
    }
    if (MCTruthTTClusterHandle.isValid() && MCTruthTTClusterHandle->findTTClusterRefs(tp_ptr).empty())
      continue;

    int tmp_tp_pdgid = iterTP.pdgId();

    float tmp_tp_z0, tmp_tp_Lxy, tmp_tp_d0;
    std::tie(tmp_tp_z0, tmp_tp_Lxy, tmp_tp_d0) = phase2tkutil::computeZ0LxyD0(*tp_ptr);

    if (std::fabs(tmp_tp_z0) > TP_maxZ0)
      continue;

    // To make efficiency plots where the denominator has NO stub cuts
    if (tmp_tp_Lxy < 1.0) {
      tp_pt->Fill(tmp_tp_pt);  // pT effic, no cut on pT, but Lxy cut
      if (tmp_tp_pt <= 10)
        tp_pt_zoom->Fill(tmp_tp_pt);  // pT effic, no cut on pT, but Lxy cut
    }
    tp_pt_for_dis->Fill(tmp_tp_pt);  // pT effic for displaced, no cut on pT or Lxy
    if (tmp_tp_pt <= 10) {
      tp_pt_zoom_for_dis->Fill(tmp_tp_pt);  // pT effic for displaced, no cut on pT or Lxy
    }

    if (tmp_tp_pt < TP_minPt)
      continue;
    tp_Lxy->Fill(tmp_tp_Lxy);  // Lxy efficiency has no cut on Lxy
    tp_eta_for_dis->Fill(tmp_tp_eta);
    tp_d0_for_dis->Fill(tmp_tp_d0);
    tp_z0_for_dis->Fill(tmp_tp_z0);

    if (MCTruthTTTrackExtendedHandle.isValid()) {
      if (nStubTP >= TP_minNStub || nStubLayerTP >= TP_minNLayersStub) {
        processTrackCollection(MCTruthTTTrackExtendedHandle,
                               tp_ptr,
                               true,
                               tmp_tp_pdgid,
                               tmp_tp_d0,
                               tmp_tp_Lxy,
                               tmp_tp_pt,
                               tmp_tp_eta,
                               tmp_tp_phi,
                               tmp_tp_z0,
                               iEvent);  // Extended Tracks
      }
    }

    if (tmp_tp_Lxy > TP_maxLxy)
      continue;

    tp_eta->Fill(tmp_tp_eta);
    tp_d0->Fill(tmp_tp_d0);
    tp_z0->Fill(tmp_tp_z0);

    if (nStubTP < TP_minNStub || nStubLayerTP < TP_minNLayersStub)
      continue;  // nStub cut not included in denominator of efficiency plots

    if (MCTruthTTTrackHandle.isValid()) {
      processTrackCollection(MCTruthTTTrackHandle,
                             tp_ptr,
                             false,
                             tmp_tp_pdgid,
                             tmp_tp_d0,
                             tmp_tp_Lxy,
                             tmp_tp_pt,
                             tmp_tp_eta,
                             tmp_tp_phi,
                             tmp_tp_z0,
                             iEvent);  // Regular L1 Tracks
    }
  }  // end loop over tracking particles
  n_trackParts->Fill(nTrackParts);
  nTrackParts = 0;
}  // end of method

// ------------ method called once each job just before starting event loop
// ------------
void Phase2OTValidateTracks::bookHistograms(DQMStore::IBooker &iBooker,
                                            edm::Run const &run,
                                            edm::EventSetup const &es) {
  edm::ParameterSet psTrackParts_Pt = conf_.getParameter<edm::ParameterSet>("TH1TrackParts_Pt");
  edm::ParameterSet psTrackParts_Eta = conf_.getParameter<edm::ParameterSet>("TH1TrackParts_Eta");
  edm::ParameterSet psTrackParts_Phi = conf_.getParameter<edm::ParameterSet>("TH1TrackParts_Phi");
  edm::ParameterSet psEffic_pt = conf_.getParameter<edm::ParameterSet>("TH1Effic_pt");
  edm::ParameterSet psEffic_pt_zoom = conf_.getParameter<edm::ParameterSet>("TH1Effic_pt_zoom");
  edm::ParameterSet psEffic_eta = conf_.getParameter<edm::ParameterSet>("TH1Effic_eta");
  edm::ParameterSet psEffic_d0 = conf_.getParameter<edm::ParameterSet>("TH1Effic_d0");
  edm::ParameterSet psDisEffic_d0 = conf_.getParameter<edm::ParameterSet>("TH1DisEffic_d0");
  edm::ParameterSet psEffic_Lxy = conf_.getParameter<edm::ParameterSet>("TH1Effic_Lxy");
  edm::ParameterSet psEffic_z0 = conf_.getParameter<edm::ParameterSet>("TH1Effic_z0");
  edm::ParameterSet psRes_pt = conf_.getParameter<edm::ParameterSet>("TH1Res_pt");
  edm::ParameterSet psRes_eta = conf_.getParameter<edm::ParameterSet>("TH1Res_eta");
  edm::ParameterSet psRes_ptRel = conf_.getParameter<edm::ParameterSet>("TH1Res_ptRel");
  edm::ParameterSet psRes_phi = conf_.getParameter<edm::ParameterSet>("TH1Res_phi");
  edm::ParameterSet psRes_z0 = conf_.getParameter<edm::ParameterSet>("TH1Res_z0");
  edm::ParameterSet psRes_d0 = conf_.getParameter<edm::ParameterSet>("TH1Res_d0");
  edm::ParameterSet psRes_displaced_d0 = conf_.getParameter<edm::ParameterSet>("TH1Resdisplaced_d0");
  edm::ParameterSet n_trackParticles = conf_.getParameter<edm::ParameterSet>("n_trackParticles");
  // Histogram setup and definitions
  std::string HistoName;
  iBooker.setCurrentFolder(topFolderName_ + "/trackParticles");

  // 1D: pT
  HistoName = "trackParts_Pt";
  trackParts_Pt = iBooker.book1D(HistoName,
                                 HistoName,
                                 psTrackParts_Pt.getParameter<int32_t>("Nbinsx"),
                                 psTrackParts_Pt.getParameter<double>("xmin"),
                                 psTrackParts_Pt.getParameter<double>("xmax"));
  trackParts_Pt->setAxisTitle("p_{T} [GeV]", 1);
  trackParts_Pt->setAxisTitle("# tracking particles", 2);

  // 1D: eta
  HistoName = "trackParts_Eta";
  trackParts_Eta = iBooker.book1D(HistoName,
                                  HistoName,
                                  psTrackParts_Eta.getParameter<int32_t>("Nbinsx"),
                                  psTrackParts_Eta.getParameter<double>("xmin"),
                                  psTrackParts_Eta.getParameter<double>("xmax"));
  trackParts_Eta->setAxisTitle("#eta", 1);
  trackParts_Eta->setAxisTitle("# tracking particles", 2);

  // 1D: phi
  HistoName = "trackParts_Phi";
  trackParts_Phi = iBooker.book1D(HistoName,
                                  HistoName,
                                  psTrackParts_Phi.getParameter<int32_t>("Nbinsx"),
                                  psTrackParts_Phi.getParameter<double>("xmin"),
                                  psTrackParts_Phi.getParameter<double>("xmax"));
  trackParts_Phi->setAxisTitle("#phi", 1);
  trackParts_Phi->setAxisTitle("# tracking particles", 2);

  HistoName = "n_trackParts";
  n_trackParts = iBooker.book1D(HistoName,
                                HistoName,
                                n_trackParticles.getParameter<int32_t>("Nbinsx"),
                                n_trackParticles.getParameter<double>("xmin"),
                                n_trackParticles.getParameter<double>("xmax"));
  n_trackParts->setAxisTitle("# track particles per event", 1);

  // 1D plots for nominal collection efficiency
  iBooker.setCurrentFolder(topFolderName_ + "/Nominal_L1TF/EfficiencyIngredients");
  // pT
  HistoName = "tp_pt";
  tp_pt = iBooker.book1D(HistoName,
                         HistoName,
                         psEffic_pt.getParameter<int32_t>("Nbinsx"),
                         psEffic_pt.getParameter<double>("xmin"),
                         psEffic_pt.getParameter<double>("xmax"));
  tp_pt->setAxisTitle("p_{T} [GeV]", 1);
  tp_pt->setAxisTitle("# tracking particles", 2);

  // Matched TP's pT
  HistoName = "match_tp_pt";
  match_tp_pt = iBooker.book1D(HistoName,
                               HistoName,
                               psEffic_pt.getParameter<int32_t>("Nbinsx"),
                               psEffic_pt.getParameter<double>("xmin"),
                               psEffic_pt.getParameter<double>("xmax"));
  match_tp_pt->setAxisTitle("p_{T} [GeV]", 1);
  match_tp_pt->setAxisTitle("# matched tracking particles", 2);

  // pT zoom (0-10 GeV)
  HistoName = "tp_pt_zoom";
  tp_pt_zoom = iBooker.book1D(HistoName,
                              HistoName,
                              psEffic_pt_zoom.getParameter<int32_t>("Nbinsx"),
                              psEffic_pt_zoom.getParameter<double>("xmin"),
                              psEffic_pt_zoom.getParameter<double>("xmax"));
  tp_pt_zoom->setAxisTitle("p_{T} [GeV]", 1);
  tp_pt_zoom->setAxisTitle("# tracking particles", 2);

  // Matched pT zoom (0-10 GeV)
  HistoName = "match_tp_pt_zoom";
  match_tp_pt_zoom = iBooker.book1D(HistoName,
                                    HistoName,
                                    psEffic_pt_zoom.getParameter<int32_t>("Nbinsx"),
                                    psEffic_pt_zoom.getParameter<double>("xmin"),
                                    psEffic_pt_zoom.getParameter<double>("xmax"));
  match_tp_pt_zoom->setAxisTitle("p_{T} [GeV]", 1);
  match_tp_pt_zoom->setAxisTitle("# matched tracking particles", 2);

  // eta
  HistoName = "tp_eta";
  tp_eta = iBooker.book1D(HistoName,
                          HistoName,
                          psEffic_eta.getParameter<int32_t>("Nbinsx"),
                          psEffic_eta.getParameter<double>("xmin"),
                          psEffic_eta.getParameter<double>("xmax"));
  tp_eta->setAxisTitle("#eta", 1);
  tp_eta->setAxisTitle("# tracking particles", 2);

  // Matched eta
  HistoName = "match_tp_eta";
  match_tp_eta = iBooker.book1D(HistoName,
                                HistoName,
                                psEffic_eta.getParameter<int32_t>("Nbinsx"),
                                psEffic_eta.getParameter<double>("xmin"),
                                psEffic_eta.getParameter<double>("xmax"));
  match_tp_eta->setAxisTitle("#eta", 1);
  match_tp_eta->setAxisTitle("# matched tracking particles", 2);

  // d0
  HistoName = "tp_d0";
  tp_d0 = iBooker.book1D(HistoName,
                         HistoName,
                         psEffic_d0.getParameter<int32_t>("Nbinsx"),
                         psEffic_d0.getParameter<double>("xmin"),
                         psEffic_d0.getParameter<double>("xmax"));
  tp_d0->setAxisTitle("d_{0} [cm]", 1);
  tp_d0->setAxisTitle("# tracking particles", 2);

  // Matched d0
  HistoName = "match_tp_d0";
  match_tp_d0 = iBooker.book1D(HistoName,
                               HistoName,
                               psEffic_d0.getParameter<int32_t>("Nbinsx"),
                               psEffic_d0.getParameter<double>("xmin"),
                               psEffic_d0.getParameter<double>("xmax"));
  match_tp_d0->setAxisTitle("d_{0} [cm]", 1);
  match_tp_d0->setAxisTitle("# matched tracking particles", 2);

  // Lxy (also known as vxy)
  HistoName = "tp_Lxy";
  tp_Lxy = iBooker.book1D(HistoName,
                          HistoName,
                          psEffic_Lxy.getParameter<int32_t>("Nbinsx"),
                          psEffic_Lxy.getParameter<double>("xmin"),
                          psEffic_Lxy.getParameter<double>("xmax"));
  tp_Lxy->setAxisTitle("L_{xy} [cm]", 1);
  tp_Lxy->setAxisTitle("# tracking particles", 2);

  // Matched Lxy
  HistoName = "match_tp_Lxy";
  match_tp_Lxy = iBooker.book1D(HistoName,
                                HistoName,
                                psEffic_Lxy.getParameter<int32_t>("Nbinsx"),
                                psEffic_Lxy.getParameter<double>("xmin"),
                                psEffic_Lxy.getParameter<double>("xmax"));
  match_tp_Lxy->setAxisTitle("L_{xy} [cm]", 1);
  match_tp_Lxy->setAxisTitle("# matched tracking particles", 2);

  // z0
  HistoName = "tp_z0";
  tp_z0 = iBooker.book1D(HistoName,
                         HistoName,
                         psEffic_z0.getParameter<int32_t>("Nbinsx"),
                         psEffic_z0.getParameter<double>("xmin"),
                         psEffic_z0.getParameter<double>("xmax"));
  tp_z0->setAxisTitle("z_{0} [cm]", 1);
  tp_z0->setAxisTitle("# tracking particles", 2);

  // Matched d0
  HistoName = "match_tp_z0";
  match_tp_z0 = iBooker.book1D(HistoName,
                               HistoName,
                               psEffic_z0.getParameter<int32_t>("Nbinsx"),
                               psEffic_z0.getParameter<double>("xmin"),
                               psEffic_z0.getParameter<double>("xmax"));
  match_tp_z0->setAxisTitle("z_{0} [cm]", 1);
  match_tp_z0->setAxisTitle("# matched tracking particles", 2);

  // 1D plots for residual
  iBooker.setCurrentFolder(topFolderName_ + "/Nominal_L1TF/Residual");
  // full pT
  HistoName = "res_pt";
  res_pt = iBooker.book1D(HistoName,
                          HistoName,
                          psRes_pt.getParameter<int32_t>("Nbinsx"),
                          psRes_pt.getParameter<double>("xmin"),
                          psRes_pt.getParameter<double>("xmax"));
  res_pt->setAxisTitle("p_{T} [GeV]", 1);
  res_pt->setAxisTitle("# tracking particles", 2);

  // Full eta
  HistoName = "res_eta";
  res_eta = iBooker.book1D(HistoName,
                           HistoName,
                           psRes_eta.getParameter<int32_t>("Nbinsx"),
                           psRes_eta.getParameter<double>("xmin"),
                           psRes_eta.getParameter<double>("xmax"));
  res_eta->setAxisTitle("#eta", 1);
  res_eta->setAxisTitle("# tracking particles", 2);

  // Relative pT
  HistoName = "res_ptRel";
  res_ptRel = iBooker.book1D(HistoName,
                             HistoName,
                             psRes_ptRel.getParameter<int32_t>("Nbinsx"),
                             psRes_ptRel.getParameter<double>("xmin"),
                             psRes_ptRel.getParameter<double>("xmax"));
  res_ptRel->setAxisTitle("Relative p_{T} [GeV]", 1);
  res_ptRel->setAxisTitle("# tracking particles", 2);

  HistoName = "res_d0";
  d0_res_hist = iBooker.book1D(HistoName,
                               HistoName,
                               psRes_d0.getParameter<int32_t>("Nbinsx"),
                               psRes_d0.getParameter<double>("xmin"),
                               psRes_d0.getParameter<double>("xmax"));
  d0_res_hist->setAxisTitle("trk d_{0} - tp d_{0} [cm]", 1);
  d0_res_hist->setAxisTitle("# tracking particles", 2);

  iBooker.setCurrentFolder(topFolderName_ + "/Nominal_L1TF/ResolutionIngredients");
  std::string ranges[6] = {"eta0to0p7", "eta0p7to1", "eta1to1p2", "eta1p2to1p6", "eta1p6to2", "eta2to2p4"};
  for (int i = 0; i < 6; i++) {
    // Eta parts (for resolution)
    HistoName = "reseta_" + ranges[i];
    reseta_vect[i] = iBooker.book1D(HistoName,
                                    HistoName,
                                    psRes_eta.getParameter<int32_t>("Nbinsx"),
                                    psRes_eta.getParameter<double>("xmin"),
                                    psRes_eta.getParameter<double>("xmax"));
    reseta_vect[i]->setAxisTitle("#eta_{trk} - #eta_{tp}", 1);
    reseta_vect[i]->setAxisTitle("# tracking particles", 2);

    // pT parts for resolution (pT res vs eta)
    // pT a (2 to 3 GeV)
    HistoName = "respt_" + ranges[i] + "_pt2to3";
    respt_pt2to3[i] = iBooker.book1D(HistoName,
                                     HistoName,
                                     psRes_pt.getParameter<int32_t>("Nbinsx"),
                                     psRes_pt.getParameter<double>("xmin"),
                                     psRes_pt.getParameter<double>("xmax"));
    respt_pt2to3[i]->setAxisTitle("(p_{T}(trk) - p_{T}(tp))/p_{T}(tp)", 1);
    respt_pt2to3[i]->setAxisTitle("# tracking particles", 2);

    // pT b (3 to 8 GeV)
    HistoName = "respt_" + ranges[i] + "_pt3to8";
    respt_pt3to8[i] = iBooker.book1D(HistoName,
                                     HistoName,
                                     psRes_pt.getParameter<int32_t>("Nbinsx"),
                                     psRes_pt.getParameter<double>("xmin"),
                                     psRes_pt.getParameter<double>("xmax"));
    respt_pt3to8[i]->setAxisTitle("(p_{T}(trk) - p_{T}(tp))/p_{T}(tp)", 1);
    respt_pt3to8[i]->setAxisTitle("# tracking particles", 2);

    // pT c (>8 GeV)
    HistoName = "respt_" + ranges[i] + "_pt8toInf";
    respt_pt8toInf[i] = iBooker.book1D(HistoName,
                                       HistoName,
                                       psRes_pt.getParameter<int32_t>("Nbinsx"),
                                       psRes_pt.getParameter<double>("xmin"),
                                       psRes_pt.getParameter<double>("xmax"));
    respt_pt8toInf[i]->setAxisTitle("(p_{T}(trk) - p_{T}(tp))/p_{T}(tp)", 1);
    respt_pt8toInf[i]->setAxisTitle("# tracking particles", 2);

    // Phi parts (for resolution)
    HistoName = "resphi_" + ranges[i];
    resphi_vect[i] = iBooker.book1D(HistoName,
                                    HistoName,
                                    psRes_phi.getParameter<int32_t>("Nbinsx"),
                                    psRes_phi.getParameter<double>("xmin"),
                                    psRes_phi.getParameter<double>("xmax"));
    resphi_vect[i]->setAxisTitle("#phi_{trk} - #phi_{tp}", 1);
    resphi_vect[i]->setAxisTitle("# tracking particles", 2);

    // z0 parts (for resolution)
    HistoName = "resz0_" + ranges[i];
    resz0_vect[i] = iBooker.book1D(HistoName,
                                   HistoName,
                                   psRes_z0.getParameter<int32_t>("Nbinsx"),
                                   psRes_z0.getParameter<double>("xmin"),
                                   psRes_z0.getParameter<double>("xmax"));
    resz0_vect[i]->setAxisTitle("z0_{trk} - z0_{tp} [cm]", 1);
    resz0_vect[i]->setAxisTitle("# tracking particles", 2);

    // d0 parts (for resolution)
    HistoName = "resd0_" + ranges[i];
    resd0_vect[i] = iBooker.book1D(HistoName,
                                   HistoName,
                                   psRes_d0.getParameter<int32_t>("Nbinsx"),
                                   psRes_d0.getParameter<double>("xmin"),
                                   psRes_d0.getParameter<double>("xmax"));
    resd0_vect[i]->setAxisTitle("d0_{trk} - d0_{tp} [cm]", 1);
    resd0_vect[i]->setAxisTitle("# tracking particles", 2);
  }

  iBooker.setCurrentFolder(topFolderName_ + "/Extended_L1TF/Displaced/EfficiencyIngredients");
  // tp pt for denom for displaced tracks
  HistoName = "tp_pt_for_dis";
  tp_pt_for_dis = iBooker.book1D(HistoName,
                                 HistoName,
                                 psEffic_pt.getParameter<int32_t>("Nbinsx"),
                                 psEffic_pt.getParameter<double>("xmin"),
                                 psEffic_pt.getParameter<double>("xmax"));
  tp_pt_for_dis->setAxisTitle("p_{T} [GeV]", 1);
  tp_pt_for_dis->setAxisTitle("# tracking particles", 2);

  // Matched Extended TP's pt
  HistoName = "match_displaced_tp_pt";
  match_displaced_tp_pt = iBooker.book1D(HistoName,
                                         HistoName,
                                         psEffic_pt.getParameter<int32_t>("Nbinsx"),
                                         psEffic_pt.getParameter<double>("xmin"),
                                         psEffic_pt.getParameter<double>("xmax"));
  match_displaced_tp_pt->setAxisTitle("p_{T} [GeV]", 1);
  match_displaced_tp_pt->setAxisTitle("# matched extended tracking particles", 2);

  // Zoom version
  HistoName = "tp_pt_zoom_for_dis";
  tp_pt_zoom_for_dis = iBooker.book1D(HistoName,
                                      HistoName,
                                      psEffic_pt_zoom.getParameter<int32_t>("Nbinsx"),
                                      psEffic_pt_zoom.getParameter<double>("xmin"),
                                      psEffic_pt_zoom.getParameter<double>("xmax"));
  tp_pt_zoom_for_dis->setAxisTitle("p_{T} [GeV]", 1);
  tp_pt_zoom_for_dis->setAxisTitle("# tracking particles", 2);

  // Zoom version
  HistoName = "match_displaced_tp_pt_zoom";
  match_displaced_tp_pt_zoom = iBooker.book1D(HistoName,
                                              HistoName,
                                              psEffic_pt_zoom.getParameter<int32_t>("Nbinsx"),
                                              psEffic_pt_zoom.getParameter<double>("xmin"),
                                              psEffic_pt_zoom.getParameter<double>("xmax"));
  match_displaced_tp_pt_zoom->setAxisTitle("p_{T} [GeV]", 1);
  match_displaced_tp_pt_zoom->setAxisTitle("# matched extended tracking particles", 2);

  // TP's eta for denom for displaced tracks
  HistoName = "tp_eta_for_dis";
  tp_eta_for_dis = iBooker.book1D(HistoName,
                                  HistoName,
                                  psEffic_eta.getParameter<int32_t>("Nbinsx"),
                                  psEffic_eta.getParameter<double>("xmin"),
                                  psEffic_eta.getParameter<double>("xmax"));
  tp_eta_for_dis->setAxisTitle("#eta", 1);
  tp_eta_for_dis->setAxisTitle("# tracking particles", 2);

  // Matched Extended TP's eta
  HistoName = "match_displaced_tp_eta";
  match_displaced_tp_eta = iBooker.book1D(HistoName,
                                          HistoName,
                                          psEffic_eta.getParameter<int32_t>("Nbinsx"),
                                          psEffic_eta.getParameter<double>("xmin"),
                                          psEffic_eta.getParameter<double>("xmax"));
  match_displaced_tp_eta->setAxisTitle("#eta", 1);
  match_displaced_tp_eta->setAxisTitle("# matched extended tracking particles", 2);

  // TP's d0 for denom for displaced tracks
  HistoName = "tp_d0_for_dis";
  tp_d0_for_dis = iBooker.book1D(HistoName,
                                 HistoName,
                                 psDisEffic_d0.getParameter<int32_t>("Nbinsx"),
                                 psDisEffic_d0.getParameter<double>("xmin"),
                                 psDisEffic_d0.getParameter<double>("xmax"));
  tp_d0_for_dis->setAxisTitle("d_{0} [cm]", 1);
  tp_d0_for_dis->setAxisTitle("# tracking particles", 2);

  // Matched Extended TP's d0
  HistoName = "match_displaced_tp_d0";
  match_displaced_tp_d0 = iBooker.book1D(HistoName,
                                         HistoName,
                                         psDisEffic_d0.getParameter<int32_t>("Nbinsx"),
                                         psDisEffic_d0.getParameter<double>("xmin"),
                                         psDisEffic_d0.getParameter<double>("xmax"));
  match_displaced_tp_d0->setAxisTitle("d_{0} [cm]", 1);
  match_displaced_tp_d0->setAxisTitle("# matched extended tracking particles", 2);

  // Matched Extended TP's Lxy
  HistoName = "match_displaced_tp_Lxy";
  match_displaced_tp_Lxy = iBooker.book1D(HistoName,
                                          HistoName,
                                          psEffic_Lxy.getParameter<int32_t>("Nbinsx"),
                                          psEffic_Lxy.getParameter<double>("xmin"),
                                          psEffic_Lxy.getParameter<double>("xmax"));
  match_displaced_tp_Lxy->setAxisTitle("L_{xy} [cm]", 1);
  match_displaced_tp_Lxy->setAxisTitle("# matched extended tracking particles", 2);

  // TP's z0 for denom for displaced tracks
  HistoName = "tp_z0_for_dis";
  tp_z0_for_dis = iBooker.book1D(HistoName,
                                 HistoName,
                                 psEffic_z0.getParameter<int32_t>("Nbinsx"),
                                 psEffic_z0.getParameter<double>("xmin"),
                                 psEffic_z0.getParameter<double>("xmax"));
  tp_z0_for_dis->setAxisTitle("z_{0} [cm]", 1);
  tp_z0_for_dis->setAxisTitle("# tracking particles", 2);

  // Matched Extended TP's z0
  HistoName = "match_displaced_tp_z0";
  match_displaced_tp_z0 = iBooker.book1D(HistoName,
                                         HistoName,
                                         psEffic_z0.getParameter<int32_t>("Nbinsx"),
                                         psEffic_z0.getParameter<double>("xmin"),
                                         psEffic_z0.getParameter<double>("xmax"));
  match_displaced_tp_z0->setAxisTitle("z_{0} [cm]", 1);
  match_displaced_tp_z0->setAxisTitle("# matched extended tracking particles", 2);

  iBooker.setCurrentFolder(topFolderName_ + "/Extended_L1TF/Displaced/Residual");

  HistoName = "res_displaced_d0";
  res_displaced_d0 = iBooker.book1D(HistoName,
                                    HistoName,
                                    psRes_displaced_d0.getParameter<int32_t>("Nbinsx"),
                                    psRes_displaced_d0.getParameter<double>("xmin"),
                                    psRes_displaced_d0.getParameter<double>("xmax"));
  res_displaced_d0->setAxisTitle("trk d_{0} - tp d_{0} [cm]", 1);
  res_displaced_d0->setAxisTitle("# displaced tracks", 2);

  HistoName = "res_displaced_eta";
  res_displaced_eta = iBooker.book1D(HistoName,
                                     HistoName,
                                     psRes_eta.getParameter<int32_t>("Nbinsx"),
                                     psRes_eta.getParameter<double>("xmin"),
                                     psRes_eta.getParameter<double>("xmax"));
  res_displaced_eta->setAxisTitle("#eta_{trk} - #eta_{tp}", 1);
  res_displaced_eta->setAxisTitle("# tracking particles", 2);

  HistoName = "res_displaced_pt";
  res_displaced_pt = iBooker.book1D(HistoName,
                                    HistoName,
                                    psRes_pt.getParameter<int32_t>("Nbinsx"),
                                    psRes_pt.getParameter<double>("xmin"),
                                    psRes_pt.getParameter<double>("xmax"));
  res_displaced_pt->setAxisTitle("p_{T}(trk)-p_{T}(tp)", 1);
  res_displaced_pt->setAxisTitle("# tracking particles", 2);

  HistoName = "res_displaced_ptRel";
  res_displaced_ptRel = iBooker.book1D(HistoName,
                                       HistoName,
                                       psRes_ptRel.getParameter<int32_t>("Nbinsx"),
                                       psRes_ptRel.getParameter<double>("xmin"),
                                       psRes_ptRel.getParameter<double>("xmax"));
  res_displaced_ptRel->setAxisTitle("Relative p_{T} [GeV]", 1);
  res_displaced_ptRel->setAxisTitle("# tracking particles", 2);

  iBooker.setCurrentFolder(topFolderName_ + "/Extended_L1TF/Displaced/ResolutionIngredients");

  for (int i = 0; i < 6; i++) {
    // reseta_displaced_vect[i]
    HistoName = "reseta_displaced_" + ranges[i];
    reseta_displaced_vect[i] = iBooker.book1D(HistoName,
                                              HistoName,
                                              psRes_eta.getParameter<int32_t>("Nbinsx"),
                                              psRes_eta.getParameter<double>("xmin"),
                                              psRes_eta.getParameter<double>("xmax"));
    reseta_displaced_vect[i]->setAxisTitle("#eta_{trk} - #eta_{tp}", 1);
    reseta_displaced_vect[i]->setAxisTitle("# tracking particles", 2);

    // resphi_displaced_vect[i]
    HistoName = "resphi_displaced_" + ranges[i];
    resphi_displaced_vect[i] = iBooker.book1D(HistoName,
                                              HistoName,
                                              psRes_phi.getParameter<int32_t>("Nbinsx"),
                                              psRes_phi.getParameter<double>("xmin"),
                                              psRes_phi.getParameter<double>("xmax"));
    resphi_displaced_vect[i]->setAxisTitle("#phi_{trk} - #phi_{tp}", 1);
    resphi_displaced_vect[i]->setAxisTitle("# tracking particles", 2);

    // resd0_displaced_vect[i]
    HistoName = "resd0_displaced_" + ranges[i];
    resd0_displaced_vect[i] = iBooker.book1D(HistoName,
                                             HistoName,
                                             psRes_displaced_d0.getParameter<int32_t>("Nbinsx"),
                                             psRes_displaced_d0.getParameter<double>("xmin"),
                                             psRes_displaced_d0.getParameter<double>("xmax"));
    resd0_displaced_vect[i]->setAxisTitle("d0_{trk} - d0_{tp} [cm]", 1);
    resd0_displaced_vect[i]->setAxisTitle("# tracking particles", 2);

    HistoName = "resz0_displaced_" + ranges[i];
    resz0_displaced_vect[i] = iBooker.book1D(HistoName,
                                             HistoName,
                                             psRes_z0.getParameter<int32_t>("Nbinsx"),
                                             psRes_z0.getParameter<double>("xmin"),
                                             psRes_z0.getParameter<double>("xmax"));
    resz0_displaced_vect[i]->setAxisTitle("z0_{trk} - z0_{tp} [cm]", 1);
    resz0_displaced_vect[i]->setAxisTitle("# tracking particles", 2);

    // pT resolution in bins (2‐3, 3‐8, >8 GeV)
    HistoName = "respt_displaced_" + ranges[i] + "_pt2to3";
    respt_displaced_pt2to3[i] = iBooker.book1D(HistoName,
                                               HistoName,
                                               psRes_pt.getParameter<int32_t>("Nbinsx"),
                                               psRes_pt.getParameter<double>("xmin"),
                                               psRes_pt.getParameter<double>("xmax"));
    respt_displaced_pt2to3[i]->setAxisTitle("p_{T}(trk)-p_{T}(tp)", 1);
    respt_displaced_pt2to3[i]->setAxisTitle("# tracking particles", 2);

    HistoName = "respt_displaced_" + ranges[i] + "_pt3to8";
    respt_displaced_pt3to8[i] = iBooker.book1D(HistoName,
                                               HistoName,
                                               psRes_pt.getParameter<int32_t>("Nbinsx"),
                                               psRes_pt.getParameter<double>("xmin"),
                                               psRes_pt.getParameter<double>("xmax"));
    respt_displaced_pt3to8[i]->setAxisTitle("p_{T}(trk)-p_{T}(tp)", 1);
    respt_displaced_pt3to8[i]->setAxisTitle("# tracking particles", 2);

    HistoName = "respt_displaced_" + ranges[i] + "_pt8toInf";
    respt_displaced_pt8toInf[i] = iBooker.book1D(HistoName,
                                                 HistoName,
                                                 psRes_pt.getParameter<int32_t>("Nbinsx"),
                                                 psRes_pt.getParameter<double>("xmin"),
                                                 psRes_pt.getParameter<double>("xmax"));
    respt_displaced_pt8toInf[i]->setAxisTitle("p_{T}(trk)-p_{T}(tp)", 1);
    respt_displaced_pt8toInf[i]->setAxisTitle("# tracking particles", 2);
  }

  iBooker.setCurrentFolder(topFolderName_ + "/Extended_L1TF/Prompt/EfficiencyIngredients");

  // Matched prompt TPs that come from the extended collection
  HistoName = "match_prompt_tp_pt";
  match_prompt_tp_pt = iBooker.book1D(HistoName,
                                      HistoName,
                                      psEffic_pt.getParameter<int32_t>("Nbinsx"),
                                      psEffic_pt.getParameter<double>("xmin"),
                                      psEffic_pt.getParameter<double>("xmax"));
  match_prompt_tp_pt->setAxisTitle("p_{T} [GeV]", 1);
  match_prompt_tp_pt->setAxisTitle("# matched tracking particles", 2);

  // Zoom version
  HistoName = "match_prompt_tp_pt_zoom";
  match_prompt_tp_pt_zoom = iBooker.book1D(HistoName,
                                           HistoName,
                                           psEffic_pt_zoom.getParameter<int32_t>("Nbinsx"),
                                           psEffic_pt_zoom.getParameter<double>("xmin"),
                                           psEffic_pt_zoom.getParameter<double>("xmax"));
  match_prompt_tp_pt_zoom->setAxisTitle("p_{T} [GeV]", 1);
  match_prompt_tp_pt_zoom->setAxisTitle("# matched tracking particles", 2);

  // Matched Extended TP's eta
  HistoName = "match_prompt_tp_eta";
  match_prompt_tp_eta = iBooker.book1D(HistoName,
                                       HistoName,
                                       psEffic_eta.getParameter<int32_t>("Nbinsx"),
                                       psEffic_eta.getParameter<double>("xmin"),
                                       psEffic_eta.getParameter<double>("xmax"));
  match_prompt_tp_eta->setAxisTitle("#eta", 1);
  match_prompt_tp_eta->setAxisTitle("# matched tracking particles", 2);

  // Matched Extended TP's d0
  HistoName = "match_prompt_tp_d0";
  match_prompt_tp_d0 = iBooker.book1D(HistoName,
                                      HistoName,
                                      psEffic_d0.getParameter<int32_t>("Nbinsx"),
                                      psEffic_d0.getParameter<double>("xmin"),
                                      psEffic_d0.getParameter<double>("xmax"));
  match_prompt_tp_d0->setAxisTitle("d_{0} [cm]", 1);
  match_prompt_tp_d0->setAxisTitle("# matched tracking particles", 2);

  // Matched Extended TP's Lxy
  HistoName = "match_prompt_tp_Lxy";
  match_prompt_tp_Lxy = iBooker.book1D(HistoName,
                                       HistoName,
                                       psEffic_Lxy.getParameter<int32_t>("Nbinsx"),
                                       psEffic_Lxy.getParameter<double>("xmin"),
                                       psEffic_Lxy.getParameter<double>("xmax"));
  match_prompt_tp_Lxy->setAxisTitle("L_{xy} [cm]", 1);
  match_prompt_tp_Lxy->setAxisTitle("# matched tracking particles", 2);

  // Matched Extended TP's z0
  HistoName = "match_prompt_tp_z0";
  match_prompt_tp_z0 = iBooker.book1D(HistoName,
                                      HistoName,
                                      psEffic_z0.getParameter<int32_t>("Nbinsx"),
                                      psEffic_z0.getParameter<double>("xmin"),
                                      psEffic_z0.getParameter<double>("xmax"));
  match_prompt_tp_z0->setAxisTitle("z_{0} [cm]", 1);
  match_prompt_tp_z0->setAxisTitle("# matched tracking particles", 2);

  iBooker.setCurrentFolder(topFolderName_ + "/Extended_L1TF/Prompt/Residual");

  HistoName = "res_prompt_d0";
  res_prompt_d0 = iBooker.book1D(HistoName,
                                 HistoName,
                                 psRes_d0.getParameter<int32_t>("Nbinsx"),
                                 psRes_d0.getParameter<double>("xmin"),
                                 psRes_d0.getParameter<double>("xmax"));
  res_prompt_d0->setAxisTitle("trk d_{0} - tp d_{0} [cm]", 1);
  res_prompt_d0->setAxisTitle("# tracking particles", 2);

  HistoName = "res_prompt_eta";
  res_prompt_eta = iBooker.book1D(HistoName,
                                  HistoName,
                                  psRes_eta.getParameter<int32_t>("Nbinsx"),
                                  psRes_eta.getParameter<double>("xmin"),
                                  psRes_eta.getParameter<double>("xmax"));
  res_prompt_eta->setAxisTitle("#eta_{trk} - #eta_{tp}", 1);
  res_prompt_eta->setAxisTitle("# tracking particles", 2);

  HistoName = "res_prompt_pt";
  res_prompt_pt = iBooker.book1D(HistoName,
                                 HistoName,
                                 psRes_pt.getParameter<int32_t>("Nbinsx"),
                                 psRes_pt.getParameter<double>("xmin"),
                                 psRes_pt.getParameter<double>("xmax"));
  res_prompt_pt->setAxisTitle("p_{T}(trk)-p_{T}(tp)", 1);
  res_prompt_pt->setAxisTitle("# tracking particles", 2);

  HistoName = "res_prompt_ptRel";
  res_prompt_ptRel = iBooker.book1D(HistoName,
                                    HistoName,
                                    psRes_ptRel.getParameter<int32_t>("Nbinsx"),
                                    psRes_ptRel.getParameter<double>("xmin"),
                                    psRes_ptRel.getParameter<double>("xmax"));
  res_prompt_ptRel->setAxisTitle("Relative p_{T} [GeV]", 1);
  res_prompt_ptRel->setAxisTitle("# tracking particles", 2);

  iBooker.setCurrentFolder(topFolderName_ + "/Extended_L1TF/Prompt/ResolutionIngredients");

  for (int i = 0; i < 6; i++) {
    // reseta_prompt_vect[i]
    HistoName = "reseta_prompt_" + ranges[i];
    reseta_prompt_vect[i] = iBooker.book1D(HistoName,
                                           HistoName,
                                           psRes_eta.getParameter<int32_t>("Nbinsx"),
                                           psRes_eta.getParameter<double>("xmin"),
                                           psRes_eta.getParameter<double>("xmax"));
    reseta_prompt_vect[i]->setAxisTitle("#eta_{trk} - #eta_{tp}", 1);
    reseta_prompt_vect[i]->setAxisTitle("# tracking particles", 2);

    // resphi_prompt_vect[i]
    HistoName = "resphi_prompt_" + ranges[i];
    resphi_prompt_vect[i] = iBooker.book1D(HistoName,
                                           HistoName,
                                           psRes_phi.getParameter<int32_t>("Nbinsx"),
                                           psRes_phi.getParameter<double>("xmin"),
                                           psRes_phi.getParameter<double>("xmax"));
    resphi_prompt_vect[i]->setAxisTitle("#phi_{trk} - #phi_{tp}", 1);
    resphi_prompt_vect[i]->setAxisTitle("# tracking particles", 2);

    // resd0_prompt_vect[i]
    HistoName = "resd0_prompt_" + ranges[i];
    resd0_prompt_vect[i] = iBooker.book1D(HistoName,
                                          HistoName,
                                          psRes_d0.getParameter<int32_t>("Nbinsx"),
                                          psRes_d0.getParameter<double>("xmin"),
                                          psRes_d0.getParameter<double>("xmax"));
    resd0_prompt_vect[i]->setAxisTitle("d0_{trk} - d0_{tp} [cm]", 1);
    resd0_prompt_vect[i]->setAxisTitle("# tracking particles", 2);

    HistoName = "resz0_prompt_" + ranges[i];
    resz0_prompt_vect[i] = iBooker.book1D(HistoName,
                                          HistoName,
                                          psRes_z0.getParameter<int32_t>("Nbinsx"),
                                          psRes_z0.getParameter<double>("xmin"),
                                          psRes_z0.getParameter<double>("xmax"));
    resz0_prompt_vect[i]->setAxisTitle("z0_{trk} - z0_{tp} [cm]", 1);
    resz0_prompt_vect[i]->setAxisTitle("# tracking particles", 2);

    // pT resolution in bins (2‐3, 3‐8, >8 GeV)
    HistoName = "respt_prompt_" + ranges[i] + "_pt2to3";
    respt_prompt_pt2to3[i] = iBooker.book1D(HistoName,
                                            HistoName,
                                            psRes_pt.getParameter<int32_t>("Nbinsx"),
                                            psRes_pt.getParameter<double>("xmin"),
                                            psRes_pt.getParameter<double>("xmax"));
    respt_prompt_pt2to3[i]->setAxisTitle("p_{T}(trk)-p_{T}(tp)", 1);
    respt_prompt_pt2to3[i]->setAxisTitle("# tracking particles", 2);

    HistoName = "respt_prompt_" + ranges[i] + "_pt3to8";
    respt_prompt_pt3to8[i] = iBooker.book1D(HistoName,
                                            HistoName,
                                            psRes_pt.getParameter<int32_t>("Nbinsx"),
                                            psRes_pt.getParameter<double>("xmin"),
                                            psRes_pt.getParameter<double>("xmax"));
    respt_prompt_pt3to8[i]->setAxisTitle("p_{T}(trk)-p_{T}(tp)", 1);
    respt_prompt_pt3to8[i]->setAxisTitle("# tracking particles", 2);

    HistoName = "respt_prompt_" + ranges[i] + "_pt8toInf";
    respt_prompt_pt8toInf[i] = iBooker.book1D(HistoName,
                                              HistoName,
                                              psRes_pt.getParameter<int32_t>("Nbinsx"),
                                              psRes_pt.getParameter<double>("xmin"),
                                              psRes_pt.getParameter<double>("xmax"));
    respt_prompt_pt8toInf[i]->setAxisTitle("p_{T}(trk)-p_{T}(tp)", 1);
    respt_prompt_pt8toInf[i]->setAxisTitle("# tracking particles", 2);
  }

}  // end of method

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
void Phase2OTValidateTracks::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  // OuterTrackerMonitorTrackingParticles
  edm::ParameterSetDescription desc;
  {
    edm::ParameterSetDescription psd0;
    psd0.add<int>("Nbinsx", 45);
    psd0.add<double>("xmax", 3);
    psd0.add<double>("xmin", -3);
    desc.add<edm::ParameterSetDescription>("TH1TrackParts_Eta", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<int>("Nbinsx", 60);
    psd0.add<double>("xmax", 3.141592653589793);
    psd0.add<double>("xmin", -3.141592653589793);
    desc.add<edm::ParameterSetDescription>("TH1TrackParts_Phi", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<int>("Nbinsx", 45);
    psd0.add<double>("xmax", 100);
    psd0.add<double>("xmin", 0);
    desc.add<edm::ParameterSetDescription>("TH1TrackParts_Pt", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<int>("Nbinsx", 200);
    psd0.add<double>("xmax", 0.5);
    psd0.add<double>("xmin", -0.5);
    desc.add<edm::ParameterSetDescription>("TH1Res_ptRel", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<int>("Nbinsx", 50);
    psd0.add<double>("xmax", 100);
    psd0.add<double>("xmin", 0);
    desc.add<edm::ParameterSetDescription>("TH1Effic_pt", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<int>("Nbinsx", 50);
    psd0.add<double>("xmax", 10);
    psd0.add<double>("xmin", 0);
    desc.add<edm::ParameterSetDescription>("TH1Effic_pt_zoom", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<int>("Nbinsx", 50);
    psd0.add<double>("xmax", 2.5);
    psd0.add<double>("xmin", -2.5);
    desc.add<edm::ParameterSetDescription>("TH1Effic_eta", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<int>("Nbinsx", 101);
    psd0.add<double>("xmax", 0.15);
    psd0.add<double>("xmin", -0.15);
    desc.add<edm::ParameterSetDescription>("TH1Effic_d0", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<int>("Nbinsx", 101);
    psd0.add<double>("xmax", 10);
    psd0.add<double>("xmin", -10);
    desc.add<edm::ParameterSetDescription>("TH1DisEffic_d0", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<int>("Nbinsx", 25);
    psd0.add<double>("xmax", 1.0);
    psd0.add<double>("xmin", 0);
    desc.add<edm::ParameterSetDescription>("TH1Effic_Lxy", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<int>("Nbinsx", 40);
    psd0.add<double>("xmax", 16);
    psd0.add<double>("xmin", -16);
    desc.add<edm::ParameterSetDescription>("TH1Effic_z0", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<int>("Nbinsx", 100);
    psd0.add<double>("xmax", 0.2);
    psd0.add<double>("xmin", -0.2);
    desc.add<edm::ParameterSetDescription>("TH1Res_pt", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<int>("Nbinsx", 100);
    psd0.add<double>("xmax", 0.01);
    psd0.add<double>("xmin", -0.01);
    desc.add<edm::ParameterSetDescription>("TH1Res_eta", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<int>("Nbinsx", 100);
    psd0.add<double>("xmax", 0.01);
    psd0.add<double>("xmin", -0.01);
    desc.add<edm::ParameterSetDescription>("TH1Res_phi", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<int>("Nbinsx", 100);
    psd0.add<double>("xmax", 1.0);
    psd0.add<double>("xmin", -1.0);
    desc.add<edm::ParameterSetDescription>("TH1Res_z0", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<int>("Nbinsx", 100);
    psd0.add<double>("xmax", 0.05);
    psd0.add<double>("xmin", -0.05);
    desc.add<edm::ParameterSetDescription>("TH1Res_d0", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<int>("Nbinsx", 101);
    psd0.add<double>("xmax", 2.0);
    psd0.add<double>("xmin", -2.0);
    desc.add<edm::ParameterSetDescription>("TH1Resdisplaced_d0", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<int>("Nbinsx", 100);
    psd0.add<double>("xmax", 600.0);
    psd0.add<double>("xmin", 0.0);
    desc.add<edm::ParameterSetDescription>("n_trackParticles", psd0);
  }
  desc.add<std::string>("TopFolderName", "TrackerPhase2OTL1TrackV");
  desc.add<edm::InputTag>("trackingParticleToken", edm::InputTag("mix", "MergedTrackTruth"));
  desc.add<edm::InputTag>("MCTruthStubInputTag", edm::InputTag("TTStubAssociatorFromPixelDigis", "StubAccepted"));
  desc.add<edm::InputTag>("MCTruthTrackInputTag", edm::InputTag("TTTrackAssociatorFromPixelDigis", "Level1TTTracks"));
  desc.add<edm::InputTag>("MCTruthTrackExtendedInputTag",
                          edm::InputTag("TTTrackAssociatorFromPixelDigisExtended", "Level1TTTracks"));
  desc.add<edm::InputTag>("MCTruthClusterInputTag",
                          edm::InputTag("TTClusterAssociatorFromPixelDigis", "ClusterInclusive"));
  desc.add<int>("L1Tk_minNStub", 4);
  desc.add<double>("L1Tk_maxChi2dof", 25.0);
  desc.add<int>("TP_minNStub", 4);
  desc.add<int>("TP_minNLayersStub", 4);
  desc.add<double>("TP_minPt", 1.5);
  desc.add<double>("TP_maxEta", 2.4);
  desc.add<double>("TP_maxZ0", 15.0);
  desc.add<double>("TP_maxLxy", 1.0);
  desc.add<double>("TP_maxD0", 0.1);
  descriptions.add("Phase2OTValidateTracks", desc);
  // or use the following to generate the label from the module's C++ type
  // descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(Phase2OTValidateTracks);
