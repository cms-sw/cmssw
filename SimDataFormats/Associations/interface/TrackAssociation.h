#ifndef TrackAssociation_h
#define TrackAssociation_h

#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "DataFormats/Common/interface/OneToManyWithQualityGeneric.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/Common/interface/View.h"

namespace reco {

  template <typename T_TrackColl>
  using SimToRecoCollectionT =
      edm::AssociationMap<edm::OneToManyWithQualityGeneric<TrackingParticleCollection, T_TrackColl, double>>;

  using SimToRecoCollection = SimToRecoCollectionT<edm::View<reco::Track>>;
  using SimToRecoCollectionSeed = SimToRecoCollectionT<edm::View<TrajectorySeed>>;
  using SimToRecoCollectionTCandidate = SimToRecoCollectionT<TrackCandidateCollection>;

  template <typename T_TrackColl>
  using RecoToSimCollectionT =
      edm::AssociationMap<edm::OneToManyWithQualityGeneric<T_TrackColl, TrackingParticleCollection, double>>;

  using RecoToSimCollection = RecoToSimCollectionT<edm::View<reco::Track>>;
  using RecoToSimCollectionSeed = RecoToSimCollectionT<edm::View<TrajectorySeed>>;
  using RecoToSimCollectionTCandidate = RecoToSimCollectionT<TrackCandidateCollection>;

}  // namespace reco

#endif
