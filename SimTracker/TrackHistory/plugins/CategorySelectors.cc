
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "PhysicsTools/UtilAlgos/interface/ObjectSelector.h"

#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"

#include "SimTracker/TrackHistory/interface/CategoryCriteria.h"
#include "SimTracker/TrackHistory/interface/TrackClassifier.h"
#include "SimTracker/TrackHistory/interface/VertexClassifier.h"

namespace reco
{
namespace modules
{

// Generic TrackCategory selector

typedef ObjectSelector<CategoryCriteria<TrackCollection, TrackClassifier, TrackCategories> > TrackSelector;
DEFINE_FWK_MODULE(TrackSelector);

typedef ObjectSelector<CategoryCriteria<TrackingParticleCollection, TrackClassifier, TrackCategories> > TrackingParticleSelector;
DEFINE_FWK_MODULE(TrackingParticleSelector);

// Generic VertexCategory selector

typedef ObjectSelector<CategoryCriteria<VertexCollection, VertexClassifier, VertexCategories> > VertexSelector;
DEFINE_FWK_MODULE(VertexSelector);

typedef ObjectSelector<CategoryCriteria<TrackingVertexCollection, VertexClassifier, VertexCategories> > TrackingVertexSelector;
DEFINE_FWK_MODULE(TrackingVertexSelector);

}
}
