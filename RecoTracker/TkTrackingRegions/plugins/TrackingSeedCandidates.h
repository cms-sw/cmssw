#ifndef RecoTracker_TkTrackingRegions_TrackingSeedCandidates_h
#define RecoTracker_TkTrackingRegions_TrackingSeedCandidates_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Common/interface/Handle.h"

#include <vector>
#include <utility>

class TrackingSeedCandidates {
public:
  enum class SeedingMode { CANDIDATE_SEEDED, GLOBAL };
  //using Objects = std::pair< edm::Handle< reco::CandidateView > , std::pair < float, float > > ; // (origin, half-length in z)
  using Objects = std::pair<const reco::CandidateView*, std::pair<float, float> >;  // (origin, half-length in z)
  TrackingSeedCandidates(const edm::ParameterSet& regPSet, edm::ConsumesCollector&& iC)
      : TrackingSeedCandidates(regPSet, iC) {}
  TrackingSeedCandidates(const edm::ParameterSet& regPSet, edm::ConsumesCollector& iC);
  ~TrackingSeedCandidates() = default;

  static void fillDescriptions(edm::ParameterSetDescription& desc);

  Objects objects(const edm::Event& iEvent) const;

private:
  SeedingMode m_seedingMode;
  float m_deltaEta_Cand;
  float m_deltaPhi_Cand;

  edm::EDGetTokenT<reco::CandidateView> m_token_input;
};

#endif
