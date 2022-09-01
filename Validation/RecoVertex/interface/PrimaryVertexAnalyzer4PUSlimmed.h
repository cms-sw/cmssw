#ifndef VALIDATION_RECOVERTEX_INTERFACE_PRIMARYVERTEXANALYZER4PUSLIMMED_H_
#define VALIDATION_RECOVERTEX_INTERFACE_PRIMARYVERTEXANALYZER4PUSLIMMED_H_

// -*- C++ -*-
//
// Package:    PrimaryVertexAnalyzer4PUSlimmed
// Class:      PrimaryVertexAnalyzer4PUSlimmed
//
/**\class PrimaryVertexAnalyzer4PUSlimmed PrimaryVertexAnalyzer4PUSlimmed.cc Validation/RecoVertex/src/PrimaryVertexAnalyzer4PUSlimmed.cc

   Description: primary vertex analyzer for events with pile-up

   Implementation:
   <Notes on implementation>
*/
//
// Original Author: Marco Rovere (code adapted from old code by
// Wolfram Erdmann)

// system include files
#include <memory>
#include <string>
#include <vector>
#include <map>
#include <algorithm>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/ESHandle.h"

// math
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/Point3D.h"

// reco track
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

// reco vertex
#include "DataFormats/VertexReco/interface/VertexFwd.h"

// simulated track
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/Associations/interface/TrackToTrackingParticleAssociator.h"

// pile-up
#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"

// vertexing
#include "RecoVertex/PrimaryVertexProducer/interface/TrackFilterForPVFinding.h"

// simulated vertex
#include "SimDataFormats/Associations/interface/VertexToTrackingVertexAssociator.h"

// DQM
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"

// class declaration
class PrimaryVertexAnalyzer4PUSlimmed : public DQMEDAnalyzer {
  typedef math::XYZTLorentzVector LorentzVector;

  enum SignalVertexKind { HIGHEST_PT = 0, IS_ASSOC2FIRST_RECO = 1, IS_ASSOC2ANY_RECO = 2 };

  // auxiliary class holding simulated vertices
  struct simPrimaryVertex {
    simPrimaryVertex(double x1, double y1, double z1)
        : x(x1),
          y(y1),
          z(z1),
          ptsq(0),
          closest_vertex_distance_z(-1.),
          nGenTrk(0),
          num_matched_reco_tracks(0),
          average_match_quality(0.0) {
      ptot.setPx(0);
      ptot.setPy(0);
      ptot.setPz(0);
      ptot.setE(0);
      p4 = LorentzVector(0, 0, 0, 0);
      r = sqrt(x * x + y * y);
    };
    double x, y, z, r;
    HepMC::FourVector ptot;
    LorentzVector p4;
    double ptsq;
    double closest_vertex_distance_z;
    int nGenTrk;
    int num_matched_reco_tracks;
    float average_match_quality;
    EncodedEventId eventId;
    TrackingVertexRef sim_vertex;
    std::vector<const reco::Vertex *> rec_vertices;
  };

  // auxiliary class holding reconstructed vertices
  struct recoPrimaryVertex {
    enum VertexProperties { NONE = 0, MATCHED = 1, DUPLICATE = 2, MERGED = 4 };
    recoPrimaryVertex(double x1, double y1, double z1)
        : x(x1),
          y(y1),
          z(z1),
          pt(0),
          ptsq(0),
          closest_vertex_distance_z(-1.),
          purity(-1.),
          nRecoTrk(0),
          num_matched_sim_tracks(0),
          kind_of_vertex(0),
          recVtx(nullptr) {
      r = sqrt(x * x + y * y);
    };
    double x, y, z, r;
    double pt;
    double ptsq;
    double closest_vertex_distance_z;
    double purity;  // calculated and assigned in calculatePurityAndFillHistograms
    int nRecoTrk;
    int num_matched_sim_tracks;
    int kind_of_vertex;
    std::vector<const TrackingVertex *> sim_vertices;
    std::vector<const simPrimaryVertex *> sim_vertices_internal;
    std::vector<unsigned int> sim_vertices_num_shared_tracks;
    const reco::Vertex *recVtx;
    reco::VertexBaseRef recVtxRef;
  };

public:
  explicit PrimaryVertexAnalyzer4PUSlimmed(const edm::ParameterSet &);
  ~PrimaryVertexAnalyzer4PUSlimmed() override;

  void analyze(const edm::Event &, const edm::EventSetup &) override;
  void bookHistograms(DQMStore::IBooker &i, edm::Run const &, edm::EventSetup const &) override;

private:
  void resetSimPVAssociation(std::vector<simPrimaryVertex> &);
  void matchSim2RecoVertices(std::vector<simPrimaryVertex> &, const reco::VertexSimToRecoCollection &);
  void matchReco2SimVertices(std::vector<recoPrimaryVertex> &,
                             const reco::VertexRecoToSimCollection &,
                             const std::vector<simPrimaryVertex> &);
  bool matchRecoTrack2SimSignal(const reco::TrackBaseRef &);
  void fillGenericGenVertexHistograms(const simPrimaryVertex &v);
  // void fillGenericRecoVertexHistograms(const std::string &,
  //                                      const simPrimaryVertex &v);
  void fillRecoAssociatedGenVertexHistograms(const std::string &, const simPrimaryVertex &v);
  void fillRecoAssociatedGenPVHistograms(const std::string &label,
                                         const PrimaryVertexAnalyzer4PUSlimmed::simPrimaryVertex &v,
                                         bool genPVMatchedToRecoPV);
  void fillGenAssociatedRecoVertexHistograms(const std::string &, int, recoPrimaryVertex &v);
  void fillResolutionAndPullHistograms(const std::string &, int, recoPrimaryVertex &v, bool);

  void calculatePurityAndFillHistograms(const std::string &, std::vector<recoPrimaryVertex> &, int, bool);

  std::vector<PrimaryVertexAnalyzer4PUSlimmed::simPrimaryVertex> getSimPVs(
      const edm::Handle<TrackingVertexCollection> &);

  std::vector<PrimaryVertexAnalyzer4PUSlimmed::recoPrimaryVertex> getRecoPVs(
      const edm::Handle<edm::View<reco::Vertex>> &);

  template <class T>
  void computePairDistance(const T &collection, MonitorElement *me);

  // ----------member data ---------------------------
  bool verbose_;
  bool use_only_charged_tracks_;
  const bool do_generic_sim_plots_;
  std::string root_folder_;

  std::map<std::string, std::map<std::string, MonitorElement *>> mes_;
  const reco::RecoToSimCollection *r2s_;
  const reco::SimToRecoCollection *s2r_;

  edm::EDGetTokenT<std::vector<PileupSummaryInfo>> vecPileupSummaryInfoToken_;
  std::vector<edm::EDGetTokenT<edm::View<reco::Vertex>>> reco_vertex_collection_tokens_;
  std::vector<edm::InputTag> reco_vertex_collections_;
  edm::EDGetTokenT<TrackingParticleCollection> trackingParticleCollectionToken_;
  edm::EDGetTokenT<TrackingVertexCollection> trackingVertexCollectionToken_;
  edm::EDGetTokenT<reco::SimToRecoCollection> simToRecoAssociationToken_;
  edm::EDGetTokenT<reco::RecoToSimCollection> recoToSimAssociationToken_;
  edm::EDGetTokenT<reco::VertexToTrackingVertexAssociator> vertexAssociatorToken_;

  std::vector<bool> errorPrintedForColl_;

  unsigned int nPUbins_;
};

#endif  // VALIDATION_RECOVERTEX_INTERFACE_PRIMARYVERTEXANALYZER4PUSLIMMED_H_
