// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "TrackingTools/GsfTracking/interface/GsfTrackConstraintAssociation.h"

//
// class decleration
//

class GsfVertexConstraintProducer: public edm::EDProducer {
public:
  explicit GsfVertexConstraintProducer(const edm::ParameterSet&);
  ~GsfVertexConstraintProducer();

private:
  virtual void produce(edm::Event&, const edm::EventSetup&) override;
  virtual void endJob() ;
      
  // ----------member data ---------------------------
  const edm::ParameterSet iConfig_;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
GsfVertexConstraintProducer::GsfVertexConstraintProducer(const edm::ParameterSet& iConfig) : iConfig_(iConfig)
{
  //register your products
  produces<std::vector<VertexConstraint> >();
  produces<GsfTrackVtxConstraintAssociationCollection>();

  //now do what ever other initialization is needed
}


GsfVertexConstraintProducer::~GsfVertexConstraintProducer()
{
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void GsfVertexConstraintProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  InputTag srcTag = iConfig_.getParameter<InputTag>("src");
  Handle<reco::GsfTrackCollection> theTCollection;
  iEvent.getByLabel(srcTag,theTCollection);
  
  std::auto_ptr<std::vector<VertexConstraint> > pairs(new std::vector<VertexConstraint>);
  std::auto_ptr<GsfTrackVtxConstraintAssociationCollection> output(new GsfTrackVtxConstraintAssociationCollection);
  
  edm::RefProd<std::vector<VertexConstraint> > rPairs = iEvent.getRefBeforePut<std::vector<VertexConstraint> >();

  int index = 0;
  for (reco::GsfTrackCollection::const_iterator i=theTCollection->begin(); i!=theTCollection->end();i++) {
    VertexConstraint tmp(GlobalPoint(0,0,0),GlobalError(0.01,0,0.01,0,0,0.001));
    pairs->push_back(tmp);
    output->insert(reco::GsfTrackRef(theTCollection,index), edm::Ref<std::vector<VertexConstraint> >(rPairs,index) );
    index++;
  }
  
  iEvent.put(pairs);
  iEvent.put(output);
}

// ------------ method called once each job just after ending the event loop  ------------
void GsfVertexConstraintProducer::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(GsfVertexConstraintProducer);
