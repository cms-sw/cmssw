#include <memory>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/Common/interface/Association.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/RefToBaseVector.h"


/**
   \class   GenMatchedCandsProducerBase GenMatchedCandsProducerBase.h "Validation/Generator/plugins/GenMatchedCandsProducerBase.h"

   \brief   Plugin template to produce generator matched candidate collections

   Plugin template to produce generator matched reco::Candidate collections. The template 
   has two arguments, which have to be expanded: 

   _Object_: specifying the objects type of the output collection of which the following 
             are implemented in the plugins directory: reco::Muon, reco::GsfElectron or 
	     reco::CaloJet.

   _Match_ : specifying the object type to which the match is expected to be performed of 
             which the following are implemented in the plugins dirctory: reco::GenParticle, 
	     reco::GenJet (the latter is only reasonable for jets). 

   The template plugin expects an edm::View of _Objecs_s, which are to be matched to 
   generator information and an edm::Association as provided by the MCMatcher as descibed 
   on WorkBookMCTruthMatch. These have to be in the event in advance. 
*/


template <typename Object, typename Match>
class GenMatchedCandsProducerBase : public edm::EDProducer {

public:
  /// constructor
  explicit GenMatchedCandsProducerBase(const edm::ParameterSet& cfg);
  /// destructor
  ~GenMatchedCandsProducerBase(){};
  
private:
  /// all that needs to be done at the beginning of a run
  virtual void beginJob(){};
  /// all that needs to done during the event loop
  virtual void produce(edm::Event& event, const edm::EventSetup& setup);
  /// all that needs to be done at the end of a run
  virtual void endJob(){};

private:
  /// input collection
  edm::InputTag src_;
  /// match to generator particles
  edm::InputTag match_;
};

/// constructor
template <typename Object, typename Match>
  GenMatchedCandsProducerBase<Object, Match>::GenMatchedCandsProducerBase(const edm::ParameterSet& cfg):
  src_( cfg.getParameter<edm::InputTag>("src") ),
  match_( cfg.getParameter<edm::InputTag>("match") )
{
  produces<edm::RefToBaseVector<Object> >();
}

/// all that needs to done during the event loop
template <typename Object, typename Match>
  void GenMatchedCandsProducerBase<Object, Match>::produce(edm::Event& evt, const edm::EventSetup& setup)
{
  // recieve the input collection
  edm::Handle<edm::View<Object> > src; 
  evt.getByLabel(src_, src);

  // recieve the match association to the generator particles
  edm::Handle<edm::Association<std::vector<Match> > > match; 
  evt.getByLabel(match_, match);
  
  // setup the output collection
  std::auto_ptr<edm::RefToBaseVector<Object> > output(new edm::RefToBaseVector<Object>);

  // iterate input collection 
  for (typename edm::View<Object>::const_iterator it = src->begin(); it != src->end(); ++it){
    unsigned int idx = it-src->begin();
    edm::RefToBase<Object> recRef = src->refAt(idx);
    if( match->contains(recRef.id()) ){
      if ((*match)[recRef].isNonnull() && (*match)[recRef].isAvailable()) {
	output->push_back(recRef);
      }
    }
  }
  // push the output
  evt.put(output);
}
