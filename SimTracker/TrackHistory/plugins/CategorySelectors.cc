#include "CommonTools/UtilAlgos/interface/ObjectSelector.h"

#include "DataFormats/BTauReco/interface/SecondaryVertexTagInfo.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "FWCore/Framework/interface/MakerMacros.h"

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

typedef ObjectSelector<CategoryCriteria<TrackCollection, TrackClassifier> > TrackCategorySelector;
DEFINE_FWK_MODULE(TrackCategorySelector);

typedef ObjectSelector<CategoryCriteria<TrackingParticleCollection, TrackClassifier> > TrackingParticleCategorySelector;
DEFINE_FWK_MODULE(TrackingParticleCategorySelector);

// Generic VertexCategory selector

typedef ObjectSelector<CategoryCriteria<VertexCollection, VertexClassifier> > VertexCategorySelector;
DEFINE_FWK_MODULE(VertexCategorySelector);

typedef ObjectSelector<CategoryCriteria<TrackingVertexCollection, VertexClassifier> > TrackingVertexCategorySelector;
DEFINE_FWK_MODULE(TrackingVertexCategorySelector);

typedef ObjectSelector<
CategoryCriteria<SecondaryVertexTagInfoCollection, VertexClassifierByProxy<SecondaryVertexTagInfoCollection> >
> SecondaryVertexTagInfoCategorySelector;
DEFINE_FWK_MODULE(SecondaryVertexTagInfoCategorySelector);

}
}
