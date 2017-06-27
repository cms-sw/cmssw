// -*- C++ -*-
//
// Package:    V0Validator
// Class:      V0Validator
//
/**\class V0Validator V0Validator.cc
 Validation/RecoVertex/interface/V0Validator.h

 Description: Creates validation histograms for RecoVertex/V0Producer

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Brian Drell
//         Created:  Wed Feb 18 17:21:04 MST 2009
//
//

// system include files
#include <memory>
#include <array>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertex.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/PatternTools/interface/TrajectoryStateClosestToBeamLineBuilder.h"
#include "TrackingTools/PatternTools/interface/ClosestApproachInRPhi.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/VertexCompositeCandidate.h"
#include "DataFormats/V0Candidate/interface/V0Candidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"

#include "SimTracker/TrackHistory/interface/TrackClassifier.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"

#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/CommonDetUnit/interface/GluedGeomDet.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "MagneticField/VolumeBasedEngine/interface/VolumeBasedMagneticField.h"

#include "RecoVertex/VertexTools/interface/VertexDistance3D.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "HepMC/GenVertex.h"
#include "HepMC/GenParticle.h"

#include "TROOT.h"
#include "TMath.h"
#include "TH1F.h"
#include "TH1I.h"
#include "TH2F.h"

class V0Validator : public DQMEDAnalyzer {
 public:
  explicit V0Validator(const edm::ParameterSet &);
  ~V0Validator();
  enum V0Type { KSHORT, LAMBDA };
  struct V0Couple {
    reco::TrackRef one;
    reco::TrackRef two;
    explicit V0Couple(reco::TrackRef first_daughter,
                      reco::TrackRef second_daughter) {
      one = first_daughter.key() < second_daughter.key() ? first_daughter
                                                         : second_daughter;
      two = first_daughter.key() > second_daughter.key() ? first_daughter
                                                         : second_daughter;
      assert(one != two);
    }
    bool operator<(const V0Couple &rh) const {
      return one.key() < rh.one.key();
    }
    bool operator==(const V0Couple &rh) const {
      return ((one.key() == rh.one.key()) && (two.key() == rh.two.key()));
    }
  };

 private:
  virtual void analyze(const edm::Event &, const edm::EventSetup &) override;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &,
                      edm::EventSetup const &) override;
  void doFakeRates(const reco::VertexCompositeCandidateCollection &collection,
                   const reco::RecoToSimCollection &recotosimCollection,
                   V0Type t, int particle_pdgid,
                   int misreconstructed_particle_pdgid);

  void doEfficiencies(
      const TrackingVertexCollection &gen_vertices, V0Type t,
      int parent_particle_id,
      int first_daughter_id,  /* give only positive charge */
      int second_daughter_id, /* give only positive charge */
      const reco::VertexCompositeCandidateCollection &collection,
      const reco::SimToRecoCollection &simtorecoCollection);

  // MonitorElements for final histograms

  std::array<MonitorElement *, 2> candidateEffVsR_num_;
  std::array<MonitorElement *, 2> candidateEffVsEta_num_;
  std::array<MonitorElement *, 2> candidateEffVsPt_num_;
  std::array<MonitorElement *, 2> candidateTkEffVsR_num_;
  std::array<MonitorElement *, 2> candidateTkEffVsEta_num_;
  std::array<MonitorElement *, 2> candidateTkEffVsPt_num_;
  std::array<MonitorElement *, 2> candidateFakeVsR_num_;
  std::array<MonitorElement *, 2> candidateFakeVsEta_num_;
  std::array<MonitorElement *, 2> candidateFakeVsPt_num_;
  std::array<MonitorElement *, 2> candidateTkFakeVsR_num_;
  std::array<MonitorElement *, 2> candidateTkFakeVsEta_num_;
  std::array<MonitorElement *, 2> candidateTkFakeVsPt_num_;

  std::array<MonitorElement *, 2> candidateFakeVsR_denom_;
  std::array<MonitorElement *, 2> candidateFakeVsEta_denom_;
  std::array<MonitorElement *, 2> candidateFakeVsPt_denom_;
  std::array<MonitorElement *, 2> candidateEffVsR_denom_;
  std::array<MonitorElement *, 2> candidateEffVsEta_denom_;
  std::array<MonitorElement *, 2> candidateEffVsPt_denom_;

  std::array<MonitorElement *, 2> nCandidates_;
  std::array<MonitorElement *, 2> candidateStatus_;
  std::array<MonitorElement *, 2> fakeCandidateMass_;
  std::array<MonitorElement *, 2> candidateFakeDauRadDist_;
  std::array<MonitorElement *, 2> candidateMassAll;
  std::array<MonitorElement *, 2> goodCandidateMass;

  std::string theDQMRootFileName;
  std::string dirName;
  edm::EDGetTokenT<reco::RecoToSimCollection> recoRecoToSimCollectionToken_;
  edm::EDGetTokenT<reco::SimToRecoCollection> recoSimToRecoCollectionToken_;
  edm::EDGetTokenT<TrackingVertexCollection> trackingVertexCollection_Token_;
  edm::EDGetTokenT<std::vector<reco::Vertex> > vec_recoVertex_Token_;
  edm::EDGetTokenT<reco::VertexCompositeCandidateCollection>
      recoVertexCompositeCandidateCollection_k0s_Token_,
      recoVertexCompositeCandidateCollection_lambda_Token_;
};
