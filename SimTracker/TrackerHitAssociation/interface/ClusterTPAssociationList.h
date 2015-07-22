#ifndef SimTracker_TrackerHitAssociation_ClusterTPAssociationList_h
#define SimTracker_TrackerHitAssociation_ClusterTPAssociationList_h

#include "DataFormats/TrackerRecHit2D/interface/OmniClusterRef.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"

#include <vector>
#include <utility>

typedef std::vector<std::pair<OmniClusterRef, TrackingParticleRef> > ClusterTPAssociationList;

inline bool clusterTPAssociationListGreater(std::pair<OmniClusterRef, TrackingParticleRef> i,std::pair<OmniClusterRef, TrackingParticleRef> j) {
  return i.first.rawIndex() > j.first.rawIndex();
}

#endif
