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
#include "FWCore/Framework/interface/EDAnalyzer.h"
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

// DQM
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

class MonitorElement;

// class declaration
class PrimaryVertexAnalyzer4PUSlimmed : public DQMEDAnalyzer {
  typedef math::XYZTLorentzVector LorentzVector;

  enum SignalVertexKind {
    HIGHEST_PT = 0,
    IS_ASSOC2FIRST_RECO = 1,
    IS_ASSOC2ANY_RECO = 2
  };

  // auxiliary class holding simulated vertices
  struct simPrimaryVertex {
    simPrimaryVertex(double x1, double y1, double z1)
        :x(x1), y(y1), z(z1),
         ptsq(0), closest_vertex_distance_z(-1.),
         nGenTrk(0),
         num_matched_reco_tracks(0),
         average_match_quality(0.0),
         sim_vertex(nullptr) {
      ptot.setPx(0);
      ptot.setPy(0);
      ptot.setPz(0);
      ptot.setE(0);
      p4 = LorentzVector(0, 0, 0, 0);
      r = sqrt(x*x + y*y);
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
    const TrackingVertex * sim_vertex;
    std::vector<const reco::Vertex *> rec_vertices;
  };

  // auxiliary class holding reconstructed vertices
  struct recoPrimaryVertex {
    enum VertexProperties {
      NONE = 0,
      MATCHED = 1,
      DUPLICATE = 2,
      MERGED = 4
    };
    recoPrimaryVertex(double x1, double y1, double z1)
        :x(x1), y(y1), z(z1),
         ptsq(0), closest_vertex_distance_z(-1.),
         nRecoTrk(0),
         kind_of_vertex(0),
         recVtx(nullptr) {
      r = sqrt(x*x + y*y);
    };
    double x, y, z, r;
    double ptsq;
    double closest_vertex_distance_z;
    int nRecoTrk;
    int kind_of_vertex;
    std::vector<const TrackingVertex *> sim_vertices;
    std::vector<const simPrimaryVertex *> sim_vertices_internal;
    const reco::Vertex *recVtx;
  };

 public:
  explicit PrimaryVertexAnalyzer4PUSlimmed(const edm::ParameterSet&);
  ~PrimaryVertexAnalyzer4PUSlimmed();

  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void bookHistograms(DQMStore::IBooker &i,
                              edm::Run const&,
                              edm::EventSetup const&) override;

 private:
  void resetSimPVAssociation(std::vector<simPrimaryVertex>&);
  void matchSim2RecoVertices(std::vector<simPrimaryVertex>&,
                             const reco::VertexCollection &);
  void matchReco2SimVertices(std::vector<recoPrimaryVertex>&,
                             const TrackingVertexCollection &,
                             const std::vector<simPrimaryVertex>&);
  void fillGenericGenVertexHistograms(const simPrimaryVertex &v);
  // void fillGenericRecoVertexHistograms(const std::string &,
  //                                      const simPrimaryVertex &v);
  void fillRecoAssociatedGenVertexHistograms(const std::string &,
                                             const simPrimaryVertex &v);
  void fillGenAssociatedRecoVertexHistograms(const std::string &,
                                             int,
                                             recoPrimaryVertex &v);

  std::vector<PrimaryVertexAnalyzer4PUSlimmed::simPrimaryVertex> getSimPVs(
      const edm::Handle<TrackingVertexCollection>);

  std::vector<PrimaryVertexAnalyzer4PUSlimmed::recoPrimaryVertex> getRecoPVs(
      const edm::Handle<reco::VertexCollection>);

  template<class T>
  void computePairDistance(const T &collection, MonitorElement *me);

  // ----------member data ---------------------------
  bool verbose_;
  bool use_only_charged_tracks_;
  bool use_TP_associator_;
  double sigma_z_match_;
  double abs_z_match_;
  std::string root_folder_;

  std::map<std::string, std::map<std::string, MonitorElement*> > mes_;
  reco::RecoToSimCollection r2s_;
  reco::SimToRecoCollection s2r_;

  // TODO(rovere) possibly reuse an object from the event and do not
  // re-run the associator(s)
  const reco::TrackToTrackingParticleAssociator * associatorByHits_;

  edm::EDGetTokenT< std::vector<PileupSummaryInfo> > vecPileupSummaryInfoToken_;
  std::vector<edm::EDGetTokenT<reco::VertexCollection> > reco_vertex_collection_tokens_;
  std::vector<edm::InputTag > reco_vertex_collections_;
  edm::EDGetTokenT<reco::TrackCollection> recoTrackCollectionToken_;
  edm::EDGetTokenT< edm::View<reco::Track> > edmView_recoTrack_Token_;
  edm::EDGetTokenT<TrackingParticleCollection> trackingParticleCollectionToken_;
  edm::EDGetTokenT<TrackingVertexCollection> trackingVertexCollectionToken_;
  edm::EDGetTokenT<reco::TrackToTrackingParticleAssociator> recoTrackToTrackingParticleAssociatorToken_;
};

#endif  // VALIDATION_RECOVERTEX_INTERFACE_PRIMARYVERTEXANALYZER4PUSLIMMED_H_
