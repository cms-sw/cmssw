#include "TauAnalysis/MCEmbeddingTools/plugins/UniqueObjectSelector.h"

template <typename T>
UniqueObjectSelector<T>::UniqueObjectSelector(const edm::ParameterSet& cfg)
  : cut_(0),
    rank_(0)
{
  if ( cfg.existsAs<edm::InputTag>("src") ) src_.push_back(cfg.getParameter<edm::InputTag>("src"));
  else src_ = cfg.getParameter<vInputTag>("src");

  if ( cfg.exists("cut" ) ) cut_ = new StringCutObjectSelector<T>(cfg.getParameter<std::string>("cut"));
  rank_ = new StringObjectFunction<T>(cfg.getParameter<std::string>("rank"));

  filter_ = cfg.getParameter<bool>("filter");

  produces<ObjectCollection>();
}

template <typename T>
UniqueObjectSelector<T>::~UniqueObjectSelector() 
{
  delete cut_;
  delete rank_;
}

namespace
{
  template <typename T>
  struct objectWithRank
  {
    const T* object_;
    double rank_;
  };

  template <typename T>
  struct higherRankT
  {
    bool operator() (const objectWithRank<T>& t1, const objectWithRank<T>& t2)
    {
      return (t1.rank_ > t2.rank_);
    }
  };
}

template <typename T>
bool UniqueObjectSelector<T>::filter(edm::Event& evt, const edm::EventSetup& es) 
{
  std::vector<objectWithRank<T> > selectedObjects;

  // check which objects pass cuts
  for ( vInputTag::const_iterator src_i = src_.begin();
	src_i != src_.end(); ++src_i ) {
    edm::Handle<ObjectCollection> objects;
    evt.getByLabel(*src_i, objects);
    for ( typename ObjectCollection::const_iterator object = objects->begin();
	  object != objects->end(); ++object ) {
      if ( !cut_ || (*cut_)(*object) ) { // either no cut defined or object passes cut
	objectWithRank<T> selectedObject;
	selectedObject.object_ = &(*object);
	selectedObject.rank_ = (*rank_)(*object);
	selectedObjects.push_back(selectedObject);
      }
    }
  }

  // sort collection of selected objects by rank
  higherRankT<T> higherRank;
  std::sort(selectedObjects.begin(), selectedObjects.end(), higherRank);

  // store in output collection the object which passes selection and is of highest rank
  std::auto_ptr<ObjectCollection> objects_output(new ObjectCollection());
  if ( selectedObjects.size() > 0 ) objects_output->push_back(*selectedObjects.front().object_);

  evt.put(objects_output);

  if ( filter_ ) return (objects_output->size() > 0);
  else return true;
}

#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/Candidate/interface/CompositeCandidate.h"

typedef UniqueObjectSelector<pat::Muon> UniquePATMuonSelector;
typedef UniqueObjectSelector<reco::CompositeCandidate> UniqueCompositeCandidateSelector;

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(UniquePATMuonSelector);
DEFINE_FWK_MODULE(UniqueCompositeCandidateSelector);

