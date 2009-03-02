
#include "DataFormats/BTauReco/interface/SecondaryVertexTagInfo.h"
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
#include "SimTracker/TrackHistory/interface/VertexClassifierByProxy.h"

namespace reco
{
namespace modules
{

// Generic TrackCategory selector

typedef ObjectSelector<CategoryCriteria<TrackCollection, TrackClassifier> > TrackSelector;
DEFINE_FWK_MODULE(TrackSelector);

typedef ObjectSelector<CategoryCriteria<TrackingParticleCollection, TrackClassifier> > TrackingParticleSelector;
DEFINE_FWK_MODULE(TrackingParticleSelector);

// Generic VertexCategory selector

typedef ObjectSelector<CategoryCriteria<VertexCollection, VertexClassifier> > VertexSelector;
DEFINE_FWK_MODULE(VertexSelector);

typedef ObjectSelector<CategoryCriteria<TrackingVertexCollection, VertexClassifier> > TrackingVertexSelector;
DEFINE_FWK_MODULE(TrackingVertexSelector);

typedef ObjectSelector<
CategoryCriteria<SecondaryVertexTagInfoCollection, VertexClassifierByProxy<SecondaryVertexTagInfoCollection> >
> SecondaryVertexTagInfoSelector;
DEFINE_FWK_MODULE(SecondaryVertexTagInfoSelector);

}
}
