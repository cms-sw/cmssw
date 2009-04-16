// -*- C++ -*-
//
// Package:    MomentumConstraintProducer
// Class:      MomentumConstraintProducer
// 
/**\class MomentumConstraintProducer MomentumConstraintProducer.cc RecoTracker/ConstraintProducerTest/src/MomentumConstraintProducer.cc

Description: <one line class summary>

Implementation:
<Notes on implementation>
*/
//
// Original Author:  Giuseppe Cerati
//         Created:  Tue Jul 10 15:05:02 CEST 2007
// $Id: MomentumConstraintProducer.cc,v 1.4 2009/03/04 13:34:31 vlimant Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "TrackingTools/PatternTools/interface/TrackConstraintAssociation.h"

//
// class decleration
//

class MomentumConstraintProducer: public edm::EDProducer {
public:
  explicit MomentumConstraintProducer(const edm::ParameterSet&);
  ~MomentumConstraintProducer();

private:
  virtual void beginRun(edm::Run & run, const edm::EventSetup&) ;
  virtual void produce(edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
      
  // ----------member data ---------------------------
  const edm::ParameterSet iConfig_;
  double fixedmom_;
  double fixedmomerr_;
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
MomentumConstraintProducer::MomentumConstraintProducer(const edm::ParameterSet& iConfig) : iConfig_(iConfig)
{
  //register your products
  produces<std::vector<MomentumConstraint> >();
  produces<TrackMomConstraintAssociationCollection>();

  //now do what ever other initialization is needed
}


MomentumConstraintProducer::~MomentumConstraintProducer()
{
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void MomentumConstraintProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  InputTag srcTag = iConfig_.getParameter<InputTag>("src");
  fixedmom_= iConfig_.getParameter<double>("fixedMomentum");
  fixedmomerr_= iConfig_.getParameter<double>("fixedMomentumError");

  Handle<reco::TrackCollection> theTCollection;
  iEvent.getByLabel(srcTag,theTCollection);
  
  std::auto_ptr<std::vector<MomentumConstraint> > pairs(new std::vector<MomentumConstraint>);
  std::auto_ptr<TrackMomConstraintAssociationCollection> output(new TrackMomConstraintAssociationCollection);
  
  edm::RefProd<std::vector<MomentumConstraint> > rPairs = iEvent.getRefBeforePut<std::vector<MomentumConstraint> >();
  
  int index = 0;
  for (reco::TrackCollection::const_iterator i=theTCollection->begin(); i!=theTCollection->end();i++) {
    //    MomentumConstraint tmp(10.,0.01) ;
    if(fixedmom_< 0.0){
      fixedmom_= i->p();
    }
    MomentumConstraint tmp(fixedmom_,fixedmomerr_) ;
    pairs->push_back(tmp);
    output->insert(reco::TrackRef(theTCollection,index), edm::Ref<std::vector<MomentumConstraint> >(rPairs,index) );
    index++;
  }
  
  iEvent.put(pairs);
  iEvent.put(output);
}

// ------------ method called once each job just before starting event loop  ------------
void MomentumConstraintProducer::beginRun(edm::Run & run, const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void MomentumConstraintProducer::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(MomentumConstraintProducer);
