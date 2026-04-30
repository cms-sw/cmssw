
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

// Updated by: Brandi Skipworth, 2026

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
#include "Validation/SiTrackerPhase2V/interface/TrackerPhase2HistUtil.h"
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
  MonitorElement *trackParts_Num = nullptr;
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
  MonitorElement *tp_Lxy_for_dis = nullptr;      // denominator

  // extended prompt plots for denom
  MonitorElement *tp_pt_for_prompt = nullptr;       // denominator
  MonitorElement *tp_pt_zoom_for_prompt = nullptr;  // denominator
  MonitorElement *tp_eta_for_prompt = nullptr;      // denominator
  MonitorElement *tp_d0_for_prompt = nullptr;       // denominator
  MonitorElement *tp_z0_for_prompt = nullptr;       // denominator
  MonitorElement *tp_Lxy_for_prompt = nullptr;      // denominator

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
      ttTrackMCTruthExtendedToken_;  // MC truth association map for extended tracks
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
  TP_minPt = conf_.getParameter<double>("TP_minPt");                 // min pT to consider matching
  TP_maxEta = conf_.getParameter<double>("TP_maxEta");               // max eta to consider matching
  TP_maxZ0 = conf_.getParameter<double>("TP_maxZ0");                 // max vertZ (or z0) to consider matching
  TP_maxLxy = conf_.getParameter<double>("TP_maxLxy");               // max Lxy to consider prompt matching
  TP_maxD0 = conf_.getParameter<double>("TP_maxD0");                 // max d0 to consider prompt matching
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

    // ensure that track is uniquely matched to the TP we are looking at!
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
    // if extended, if "prompt"
    if (std::fabs(tmp_tp_d0) < TP_maxD0) {
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
      match_tp_Lxy_ptr = match_prompt_tp_Lxy;
      match_tp_eta_ptr = match_prompt_tp_eta;
      match_tp_d0_ptr = match_prompt_tp_d0;
      match_tp_z0_ptr = match_prompt_tp_z0;

      // if extended, if "displaced"
    } else {
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
      match_tp_Lxy_ptr = match_displaced_tp_Lxy;
      match_tp_eta_ptr = match_displaced_tp_eta;
      match_tp_d0_ptr = match_displaced_tp_d0;
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
    res_d0_hist_ptr = d0_res_hist;

    match_tp_pt_ptr = match_tp_pt;
    match_tp_pt_zoom_ptr = match_tp_pt_zoom;
    match_tp_Lxy_ptr = match_tp_Lxy;

    match_tp_eta_ptr = match_tp_eta;
    match_tp_d0_ptr = match_tp_d0;
    match_tp_z0_ptr = match_tp_z0;
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
  if (tmp_tp_Lxy < 1.0) {
    match_tp_pt_ptr->Fill(tmp_tp_pt);
    if (tmp_tp_pt > 0.f && tmp_tp_pt <= 10.f) {
      match_tp_pt_zoom_ptr->Fill(tmp_tp_pt);
    }
  }

  // Lxy efficiency (NO Lxy cut)
  match_tp_Lxy_ptr->Fill(tmp_tp_Lxy);

  // Eta, d0, z0 efficiencies
  if (isExtended && std::fabs(bestTrack_d0) >= TP_maxD0) {
    // Displaced track: Fill unconditionally (no Lxy cut for displaced)
    match_tp_eta_ptr->Fill(tmp_tp_eta);
    match_tp_d0_ptr->Fill(tmp_tp_d0);
    match_tp_z0_ptr->Fill(tmp_tp_z0);
  } else {
    // Prompt or Nominal track: Must pass Lxy cut
    if (tmp_tp_Lxy < TP_maxLxy) {
      match_tp_eta_ptr->Fill(tmp_tp_eta);
      match_tp_d0_ptr->Fill(tmp_tp_d0);
      match_tp_z0_ptr->Fill(tmp_tp_z0);
    }
  }

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

    // pT Denominators (No minPt cut yet, strict Lxy cut)
    if (tmp_tp_Lxy < 1.0) {
      tp_pt->Fill(tmp_tp_pt);
      if (tmp_tp_pt <= 10.f) {
        tp_pt_zoom->Fill(tmp_tp_pt);
      }

      if (std::fabs(tmp_tp_d0) < TP_maxD0) {
        tp_pt_for_prompt->Fill(tmp_tp_pt);
        if (tmp_tp_pt <= 10.f) {
          tp_pt_zoom_for_prompt->Fill(tmp_tp_pt);
        }
      } else {
        tp_pt_for_dis->Fill(tmp_tp_pt);
        if (tmp_tp_pt <= 10.f) {
          tp_pt_zoom_for_dis->Fill(tmp_tp_pt);
        }
      }
    } else {
      // Displaced has no Lxy cut, fill unconditionally for Lxy
      if (std::fabs(tmp_tp_d0) >= TP_maxD0) {
        tp_pt_for_dis->Fill(tmp_tp_pt);
        if (tmp_tp_pt <= 10.f) {
          tp_pt_zoom_for_dis->Fill(tmp_tp_pt);
        }
      }
    }

    if (tmp_tp_pt < TP_minPt)
      continue;

    // Lxy Denominators (minPt applied, NO Lxy cut)
    tp_Lxy->Fill(tmp_tp_Lxy);
    if (std::fabs(tmp_tp_d0) < TP_maxD0) {
      tp_Lxy_for_prompt->Fill(tmp_tp_Lxy);
    } else {
      tp_Lxy_for_dis->Fill(tmp_tp_Lxy);

      // Displaced has NO Lxy cut at all, fill its eta, d0, z0 here
      tp_eta_for_dis->Fill(tmp_tp_eta);
      tp_d0_for_dis->Fill(tmp_tp_d0);
      tp_z0_for_dis->Fill(tmp_tp_z0);
    }

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

    if (std::fabs(tmp_tp_d0) < TP_maxD0) {
      tp_eta_for_prompt->Fill(tmp_tp_eta);
      tp_d0_for_prompt->Fill(tmp_tp_d0);
      tp_z0_for_prompt->Fill(tmp_tp_z0);
    }

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
  trackParts_Num->Fill(nTrackParts);
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
  edm::ParameterSet psEffic_displaced_Lxy = conf_.getParameter<edm::ParameterSet>("TH1displacedEffic_Lxy");
  edm::ParameterSet psEffic_z0 = conf_.getParameter<edm::ParameterSet>("TH1Effic_z0");
  edm::ParameterSet psRes_pt = conf_.getParameter<edm::ParameterSet>("TH1Res_pt");
  edm::ParameterSet psRes_eta = conf_.getParameter<edm::ParameterSet>("TH1Res_eta");
  edm::ParameterSet psRes_ptRel = conf_.getParameter<edm::ParameterSet>("TH1Res_ptRel");
  edm::ParameterSet psRes_phi = conf_.getParameter<edm::ParameterSet>("TH1Res_phi");
  edm::ParameterSet psRes_z0 = conf_.getParameter<edm::ParameterSet>("TH1Res_z0");
  edm::ParameterSet psRes_d0 = conf_.getParameter<edm::ParameterSet>("TH1Res_d0");
  edm::ParameterSet psRes_displaced_d0 = conf_.getParameter<edm::ParameterSet>("TH1Resdisplaced_d0");
  edm::ParameterSet n_trackParticles = conf_.getParameter<edm::ParameterSet>("n_trackParticles");

  using phase2tkutil::book1DFromPS;

  // |eta| bin labels for resolution ingredients
  std::string ranges[6] = {"eta0to0p7", "eta0p7to1", "eta1to1p2", "eta1p2to1p6", "eta1p6to2", "eta2to2p4"};

  // Tracking particle kinematics
  iBooker.setCurrentFolder(topFolderName_ + "/trackParticles");
  trackParts_Pt = book1DFromPS(iBooker, "trackParts_Pt", psTrackParts_Pt, "p_{T} [GeV]", "# tracking particles");
  trackParts_Eta = book1DFromPS(iBooker, "trackParts_Eta", psTrackParts_Eta, "#eta", "# tracking particles");
  trackParts_Phi = book1DFromPS(iBooker, "trackParts_Phi", psTrackParts_Phi, "#phi", "# tracking particles");
  trackParts_Num =
      book1DFromPS(iBooker, "trackParts_Num", n_trackParticles, "# track particles per event", "# tracking particles");

  // Nominal L1TF: efficiency ingredients (denominator + matched numerator)
  iBooker.setCurrentFolder(topFolderName_ + "/Nominal_L1TF/EfficiencyIngredients");
  // Denominator: all selected TPs
  tp_pt = book1DFromPS(iBooker, "tp_pt", psEffic_pt, "p_{T} [GeV]", "# tracking particles");
  tp_pt_zoom = book1DFromPS(iBooker, "tp_pt_zoom", psEffic_pt_zoom, "p_{T} [GeV]", "# tracking particles");
  tp_eta = book1DFromPS(iBooker, "tp_eta", psEffic_eta, "#eta", "# tracking particles");
  tp_d0 = book1DFromPS(iBooker, "tp_d0", psEffic_d0, "d_{0} [cm]", "# tracking particles");
  tp_Lxy = book1DFromPS(iBooker, "tp_Lxy", psEffic_Lxy, "L_{xy} [cm]", "# tracking particles");
  tp_z0 = book1DFromPS(iBooker, "tp_z0", psEffic_z0, "z_{0} [cm]", "# tracking particles");

  // Numerator: matched TPs
  match_tp_pt = book1DFromPS(iBooker, "match_tp_pt", psEffic_pt, "p_{T} [GeV]", "# matched tracking particles");
  match_tp_pt_zoom =
      book1DFromPS(iBooker, "match_tp_pt_zoom", psEffic_pt_zoom, "p_{T} [GeV]", "# matched tracking particles");
  match_tp_eta = book1DFromPS(iBooker, "match_tp_eta", psEffic_eta, "#eta", "# matched tracking particles");
  match_tp_d0 = book1DFromPS(iBooker, "match_tp_d0", psEffic_d0, "d_{0} [cm]", "# matched tracking particles");
  match_tp_Lxy = book1DFromPS(iBooker, "match_tp_Lxy", psEffic_Lxy, "L_{xy} [cm]", "# matched tracking particles");
  match_tp_z0 = book1DFromPS(iBooker, "match_tp_z0", psEffic_z0, "z_{0} [cm]", "# matched tracking particles");

  // Nominal L1TF: residual distributions
  iBooker.setCurrentFolder(topFolderName_ + "/Nominal_L1TF/Residual");
  res_pt = book1DFromPS(iBooker, "res_pt", psRes_pt, "trk p_{T} - tp p_{T} [GeV]", "# tracking particles");
  res_eta = book1DFromPS(iBooker, "res_eta", psRes_eta, "trk #eta - tp #eta", "# tracking particles");
  res_ptRel = book1DFromPS(iBooker, "res_ptRel", psRes_ptRel, "Relative p_{T} [GeV]", "# tracking particles");
  d0_res_hist = book1DFromPS(iBooker, "res_d0", psRes_d0, "trk d_{0} - tp d_{0} [cm]", "# tracking particles");

  // Nominal L1TF: resolution vs eta and pT slices
  iBooker.setCurrentFolder(topFolderName_ + "/Nominal_L1TF/ResolutionIngredients");
  for (int i = 0; i < 6; i++) {
    reseta_vect[i] =
        book1DFromPS(iBooker, "reseta_" + ranges[i], psRes_eta, "#eta_{trk} - #eta_{tp}", "# tracking particles");
    resphi_vect[i] =
        book1DFromPS(iBooker, "resphi_" + ranges[i], psRes_phi, "#phi_{trk} - #phi_{tp}", "# tracking particles");
    resz0_vect[i] =
        book1DFromPS(iBooker, "resz0_" + ranges[i], psRes_z0, "z0_{trk} - z0_{tp} [cm]", "# tracking particles");
    resd0_vect[i] =
        book1DFromPS(iBooker, "resd0_" + ranges[i], psRes_d0, "d0_{trk} - d0_{tp} [cm]", "# tracking particles");
    respt_pt2to3[i] = book1DFromPS(iBooker,
                                   "respt_" + ranges[i] + "_pt2to3",
                                   psRes_pt,
                                   "(p_{T}(trk) - p_{T}(tp))/p_{T}(tp)",
                                   "# tracking particles");
    respt_pt3to8[i] = book1DFromPS(iBooker,
                                   "respt_" + ranges[i] + "_pt3to8",
                                   psRes_pt,
                                   "(p_{T}(trk) - p_{T}(tp))/p_{T}(tp)",
                                   "# tracking particles");
    respt_pt8toInf[i] = book1DFromPS(iBooker,
                                     "respt_" + ranges[i] + "_pt8toInf",
                                     psRes_pt,
                                     "(p_{T}(trk) - p_{T}(tp))/p_{T}(tp)",
                                     "# tracking particles");
  }

  // Extended L1TF (Displaced): efficiency ingredients
  iBooker.setCurrentFolder(topFolderName_ + "/Extended_L1TF/Displaced/EfficiencyIngredients");
  // Denominator: displaced TP selection
  tp_pt_for_dis = book1DFromPS(iBooker, "tp_pt_for_dis", psEffic_pt, "p_{T} [GeV]", "# tracking particles");
  tp_pt_zoom_for_dis =
      book1DFromPS(iBooker, "tp_pt_zoom_for_dis", psEffic_pt_zoom, "p_{T} [GeV]", "# tracking particles");
  tp_eta_for_dis = book1DFromPS(iBooker, "tp_eta_for_dis", psEffic_eta, "#eta", "# tracking particles");
  tp_d0_for_dis = book1DFromPS(iBooker, "tp_d0_for_dis", psDisEffic_d0, "d_{0} [cm]", "# tracking particles");
  tp_z0_for_dis = book1DFromPS(iBooker, "tp_z0_for_dis", psEffic_z0, "z_{0} [cm]", "# tracking particles");
  tp_Lxy_for_dis =
      book1DFromPS(iBooker, "tp_Lxy_for_dis", psEffic_displaced_Lxy, "L_{xy} [cm]", "# tracking particles");

  // Numerator: matched displaced TPs
  match_displaced_tp_pt = book1DFromPS(
      iBooker, "match_displaced_tp_pt", psEffic_pt, "p_{T} [GeV]", "# matched extended tracking particles");
  match_displaced_tp_pt_zoom = book1DFromPS(
      iBooker, "match_displaced_tp_pt_zoom", psEffic_pt_zoom, "p_{T} [GeV]", "# matched extended tracking particles");
  match_displaced_tp_eta =
      book1DFromPS(iBooker, "match_displaced_tp_eta", psEffic_eta, "#eta", "# matched extended tracking particles");
  match_displaced_tp_d0 = book1DFromPS(
      iBooker, "match_displaced_tp_d0", psDisEffic_d0, "d_{0} [cm]", "# matched extended tracking particles");
  match_displaced_tp_Lxy = book1DFromPS(
      iBooker, "match_displaced_tp_Lxy", psEffic_displaced_Lxy, "L_{xy} [cm]", "# matched extended tracking particles");
  match_displaced_tp_z0 =
      book1DFromPS(iBooker, "match_displaced_tp_z0", psEffic_z0, "z_{0} [cm]", "# matched extended tracking particles");

  // Extended L1TF (Displaced): residual distributions
  iBooker.setCurrentFolder(topFolderName_ + "/Extended_L1TF/Displaced/Residual");
  res_displaced_d0 =
      book1DFromPS(iBooker, "res_displaced_d0", psRes_displaced_d0, "trk d_{0} - tp d_{0} [cm]", "# displaced tracks");
  res_displaced_eta =
      book1DFromPS(iBooker, "res_displaced_eta", psRes_eta, "#eta_{trk} - #eta_{tp}", "# tracking particles");
  res_displaced_pt =
      book1DFromPS(iBooker, "res_displaced_pt", psRes_pt, "p_{T}(trk)-p_{T}(tp)", "# tracking particles");
  res_displaced_ptRel =
      book1DFromPS(iBooker, "res_displaced_ptRel", psRes_ptRel, "Relative p_{T} [GeV]", "# tracking particles");

  // Extended L1TF (Displaced): resolution vs eta and pT slices
  iBooker.setCurrentFolder(topFolderName_ + "/Extended_L1TF/Displaced/ResolutionIngredients");
  for (int i = 0; i < 6; i++) {
    reseta_displaced_vect[i] = book1DFromPS(
        iBooker, "reseta_displaced_" + ranges[i], psRes_eta, "#eta_{trk} - #eta_{tp}", "# tracking particles");
    resphi_displaced_vect[i] = book1DFromPS(
        iBooker, "resphi_displaced_" + ranges[i], psRes_phi, "#phi_{trk} - #phi_{tp}", "# tracking particles");
    resd0_displaced_vect[i] = book1DFromPS(
        iBooker, "resd0_displaced_" + ranges[i], psRes_displaced_d0, "d0_{trk} - d0_{tp} [cm]", "# tracking particles");
    resz0_displaced_vect[i] = book1DFromPS(
        iBooker, "resz0_displaced_" + ranges[i], psRes_z0, "z0_{trk} - z0_{tp} [cm]", "# tracking particles");
    respt_displaced_pt2to3[i] = book1DFromPS(
        iBooker, "respt_displaced_" + ranges[i] + "_pt2to3", psRes_pt, "p_{T}(trk)-p_{T}(tp)", "# tracking particles");
    respt_displaced_pt3to8[i] = book1DFromPS(
        iBooker, "respt_displaced_" + ranges[i] + "_pt3to8", psRes_pt, "p_{T}(trk)-p_{T}(tp)", "# tracking particles");
    respt_displaced_pt8toInf[i] = book1DFromPS(
        iBooker, "respt_displaced_" + ranges[i] + "_pt8toInf", psRes_pt, "p_{T}(trk)-p_{T}(tp)", "# tracking particles");
  }

  // Extended L1TF (Prompt): efficiency ingredients (matched prompt TPs)
  iBooker.setCurrentFolder(topFolderName_ + "/Extended_L1TF/Prompt/EfficiencyIngredients");

  // Denominator: prompt TP selection
  tp_pt_for_prompt = book1DFromPS(iBooker, "tp_pt_for_prompt", psEffic_pt, "p_{T} [GeV]", "# tracking particles");
  tp_pt_zoom_for_prompt =
      book1DFromPS(iBooker, "tp_pt_zoom_for_prompt", psEffic_pt_zoom, "p_{T} [GeV]", "# tracking particles");
  tp_eta_for_prompt = book1DFromPS(iBooker, "tp_eta_for_prompt", psEffic_eta, "#eta", "# tracking particles");
  tp_d0_for_prompt = book1DFromPS(iBooker, "tp_d0_for_prompt", psEffic_d0, "d_{0} [cm]", "# tracking particles");
  tp_Lxy_for_prompt = book1DFromPS(iBooker, "tp_Lxy_for_prompt", psEffic_Lxy, "L_{xy} [cm]", "# tracking particles");
  tp_z0_for_prompt = book1DFromPS(iBooker, "tp_z0_for_prompt", psEffic_z0, "z_{0} [cm]", "# tracking particles");

  // Numerator: matched prompt TPs
  match_prompt_tp_pt =
      book1DFromPS(iBooker, "match_prompt_tp_pt", psEffic_pt, "p_{T} [GeV]", "# matched tracking particles");
  match_prompt_tp_pt_zoom =
      book1DFromPS(iBooker, "match_prompt_tp_pt_zoom", psEffic_pt_zoom, "p_{T} [GeV]", "# matched tracking particles");
  match_prompt_tp_eta =
      book1DFromPS(iBooker, "match_prompt_tp_eta", psEffic_eta, "#eta", "# matched tracking particles");
  match_prompt_tp_d0 =
      book1DFromPS(iBooker, "match_prompt_tp_d0", psEffic_d0, "d_{0} [cm]", "# matched tracking particles");
  match_prompt_tp_Lxy =
      book1DFromPS(iBooker, "match_prompt_tp_Lxy", psEffic_Lxy, "L_{xy} [cm]", "# matched tracking particles");
  match_prompt_tp_z0 =
      book1DFromPS(iBooker, "match_prompt_tp_z0", psEffic_z0, "z_{0} [cm]", "# matched tracking particles");

  // Extended L1TF (Prompt): residual distributions
  iBooker.setCurrentFolder(topFolderName_ + "/Extended_L1TF/Prompt/Residual");
  res_prompt_d0 = book1DFromPS(iBooker, "res_prompt_d0", psRes_d0, "trk d_{0} - tp d_{0} [cm]", "# tracking particles");
  res_prompt_eta = book1DFromPS(iBooker, "res_prompt_eta", psRes_eta, "#eta_{trk} - #eta_{tp}", "# tracking particles");
  res_prompt_pt = book1DFromPS(iBooker, "res_prompt_pt", psRes_pt, "p_{T}(trk)-p_{T}(tp)", "# tracking particles");
  res_prompt_ptRel =
      book1DFromPS(iBooker, "res_prompt_ptRel", psRes_ptRel, "Relative p_{T} [GeV]", "# tracking particles");

  // Extended L1TF (Prompt): resolution vs eta and pT slices
  iBooker.setCurrentFolder(topFolderName_ + "/Extended_L1TF/Prompt/ResolutionIngredients");
  for (int i = 0; i < 6; i++) {
    reseta_prompt_vect[i] = book1DFromPS(
        iBooker, "reseta_prompt_" + ranges[i], psRes_eta, "#eta_{trk} - #eta_{tp}", "# tracking particles");
    resphi_prompt_vect[i] = book1DFromPS(
        iBooker, "resphi_prompt_" + ranges[i], psRes_phi, "#phi_{trk} - #phi_{tp}", "# tracking particles");
    resd0_prompt_vect[i] =
        book1DFromPS(iBooker, "resd0_prompt_" + ranges[i], psRes_d0, "d0_{trk} - d0_{tp} [cm]", "# tracking particles");
    resz0_prompt_vect[i] =
        book1DFromPS(iBooker, "resz0_prompt_" + ranges[i], psRes_z0, "z0_{trk} - z0_{tp} [cm]", "# tracking particles");
    respt_prompt_pt2to3[i] = book1DFromPS(
        iBooker, "respt_prompt_" + ranges[i] + "_pt2to3", psRes_pt, "p_{T}(trk)-p_{T}(tp)", "# tracking particles");
    respt_prompt_pt3to8[i] = book1DFromPS(
        iBooker, "respt_prompt_" + ranges[i] + "_pt3to8", psRes_pt, "p_{T}(trk)-p_{T}(tp)", "# tracking particles");
    respt_prompt_pt8toInf[i] = book1DFromPS(
        iBooker, "respt_prompt_" + ranges[i] + "_pt8toInf", psRes_pt, "p_{T}(trk)-p_{T}(tp)", "# tracking particles");
  }
}  // end of method

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
void Phase2OTValidateTracks::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  // OuterTrackerMonitorTrackingParticles
  edm::ParameterSetDescription desc;
  auto addHist = [&desc](const std::string &name, int bins, double xmin, double xmax) {
    edm::ParameterSetDescription psd;
    psd.add<int>("Nbinsx", bins);
    psd.add<double>("xmin", xmin);
    psd.add<double>("xmax", xmax);
    desc.add<edm::ParameterSetDescription>(name, psd);
  };

  // Tracking particle kinematics
  addHist("TH1TrackParts_Eta", 45, -3.0, 3.0);
  addHist("TH1TrackParts_Phi", 60, -3.141592653589793, 3.141592653589793);
  addHist("TH1TrackParts_Pt", 45, 0.0, 100.0);
  addHist("n_trackParticles", 100, 0.0, 600.0);

  // Efficiency plots
  addHist("TH1Effic_pt", 50, 0.0, 100.0);
  addHist("TH1Effic_pt_zoom", 50, 0.0, 10.0);
  addHist("TH1Effic_eta", 50, -2.5, 2.5);
  addHist("TH1Effic_d0", 101, -0.15, 0.15);
  addHist("TH1DisEffic_d0", 101, -10.0, 10.0);
  addHist("TH1Effic_Lxy", 25, 0.0, 1.0);
  addHist("TH1displacedEffic_Lxy", 50, 0.0, 10.0);
  addHist("TH1Effic_z0", 40, -16.0, 16.0);

  // Resolution plots
  addHist("TH1Res_ptRel", 200, -0.5, 0.5);
  addHist("TH1Res_pt", 100, -0.2, 0.2);
  addHist("TH1Res_eta", 100, -0.01, 0.01);
  addHist("TH1Res_phi", 100, -0.01, 0.01);
  addHist("TH1Res_z0", 100, -1.0, 1.0);
  addHist("TH1Res_d0", 100, -0.05, 0.05);
  addHist("TH1Resdisplaced_d0", 101, -2.0, 2.0);

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
