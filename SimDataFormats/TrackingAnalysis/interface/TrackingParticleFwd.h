#ifndef TrackingAnalysis_TrackingParticleFwd_h
#define TrackingAnalysis_TrackingParticleFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefProd.h"

class TrackingParticle;
typedef std::vector<TrackingParticle> TrackingParticleCollection;
typedef edm::Ref<TrackingParticleCollection> TrackingParticleRef;
typedef edm::RefVector<TrackingParticleCollection> TrackingParticleRefVector;
typedef edm::RefProd<TrackingParticleCollection> TrackingParticleRefProd;
typedef edm::RefVector<TrackingParticleCollection> TrackingParticleContainer;

std::ostream& operator<< (std::ostream& s, TrackingParticle const & tp);

#endif

