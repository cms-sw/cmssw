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
//
//


// system include files
#include <memory>

// user include files
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "TrackingTools/PatternTools/interface/TrackConstraintAssociation.h"

//
// class declaration
//

class MomentumConstraintProducer: public edm::EDProducer {
public:
  explicit MomentumConstraintProducer(const edm::ParameterSet&);
  ~MomentumConstraintProducer();

private:
  virtual void produce(edm::Event&, const edm::EventSetup&) override;
  virtual void endJob() ;

  // ----------member data ---------------------------
  const edm::InputTag srcTag_;
  const double fixedmom_;
  const double fixedmomerr_;
  edm::EDGetTokenT<reco::TrackCollection> srcToken_;
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
MomentumConstraintProducer::MomentumConstraintProducer(const edm::ParameterSet& iConfig):
  srcTag_(iConfig.getParameter<edm::InputTag>("src")),
  fixedmom_(iConfig.getParameter<double>("fixedMomentum")),
  fixedmomerr_(iConfig.getParameter<double>("fixedMomentumError"))

{
  //register your products
  produces<std::vector<MomentumConstraint>>();
  produces<TrackMomConstraintAssociationCollection>();

  //now do what ever other initialization is needed
  edm::ConsumesCollector&& iC = consumesCollector();
  srcToken_ = iC.consumes<reco::TrackCollection>(srcTag_);
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

  Handle<reco::TrackCollection> theTCollection;
  iEvent.getByToken(srcToken_,theTCollection);

  edm::RefProd<std::vector<MomentumConstraint>> rPairs = iEvent.getRefBeforePut<std::vector<MomentumConstraint>>();

  std::unique_ptr<std::vector<MomentumConstraint>> pairs(new std::vector<MomentumConstraint>);
  std::unique_ptr<TrackMomConstraintAssociationCollection> output(new TrackMomConstraintAssociationCollection(theTCollection, rPairs));

  int index = 0;
  for (reco::TrackCollection::const_iterator i=theTCollection->begin(); i!=theTCollection->end();i++) {
    //    MomentumConstraint tmp(10.,0.01) ;

    MomentumConstraint tmp(fixedmom_,fixedmomerr_) ;
    if(fixedmom_< 0.0){
      tmp= MomentumConstraint(i->p(),fixedmomerr_);
    }
    pairs->push_back(tmp);
    output->insert(reco::TrackRef(theTCollection,index), edm::Ref<std::vector<MomentumConstraint>>(rPairs,index) );
    index++;
  }

  iEvent.put(std::move(pairs));
  iEvent.put(std::move(output));

}

// ------------ method called once each job just after ending the event loop  ------------
void MomentumConstraintProducer::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(MomentumConstraintProducer);
