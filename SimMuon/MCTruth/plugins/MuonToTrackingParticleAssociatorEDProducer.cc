// -*- C++ -*-
//
// Package:    SimMuon/MCTruth
// Class:      MuonToTrackingParticleAssociatorEDProducer
// 
/**\class MuonToTrackingParticleAssociatorEDProducer MuonToTrackingParticleAssociatorEDProducer.cc SimMuon/MCTruth/plugins/MuonToTrackingParticleAssociatorEDProducer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Christopher Jones
//         Created:  Wed, 07 Jan 2015 21:30:14 GMT
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimMuon/MCTruth/interface/TrackerMuonHitExtractor.h"
#include "SimDataFormats/Associations/interface/MuonToTrackingParticleAssociator.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "MuonToTrackingParticleAssociatorByHitsImpl.h"

//
// class declaration
//

class MuonToTrackingParticleAssociatorEDProducer : public edm::stream::EDProducer<> {
public:
  explicit MuonToTrackingParticleAssociatorEDProducer(const edm::ParameterSet&);
  ~MuonToTrackingParticleAssociatorEDProducer();
  
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  
private:
  virtual void produce(edm::Event&, const edm::EventSetup&) override;
  
  // ----------member data ---------------------------
  edm::ParameterSet const config_;
  MuonAssociatorByHitsHelper helper_;
  TrackerMuonHitExtractor hitExtractor_;

  std::unique_ptr<RPCHitAssociator> rpctruth_;
  std::unique_ptr<DTHitAssociator> dttruth_;
  std::unique_ptr<CSCHitAssociator> csctruth_;
  std::unique_ptr<TrackerHitAssociator> trackertruth_;

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
MuonToTrackingParticleAssociatorEDProducer::MuonToTrackingParticleAssociatorEDProducer(const edm::ParameterSet& iConfig):
  config_(iConfig),
  helper_(iConfig),
  hitExtractor_(iConfig,consumesCollector())
{
   //register your products
   produces<reco::MuonToTrackingParticleAssociator>();

   //hack for consumes
   RPCHitAssociator rpctruth(iConfig,consumesCollector());
   DTHitAssociator dttruth(iConfig,consumesCollector());
   CSCHitAssociator cscruth(iConfig,consumesCollector());
   TrackerHitAssociator trackertruth(iConfig,consumesCollector());
}


MuonToTrackingParticleAssociatorEDProducer::~MuonToTrackingParticleAssociatorEDProducer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
MuonToTrackingParticleAssociatorEDProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   hitExtractor_.init(iEvent, iSetup);

   //Retrieve tracker topology from geometry
   edm::ESHandle<TrackerTopology> tTopoHand;
   iSetup.get<IdealGeometryRecord>().get(tTopoHand);
   const TrackerTopology *tTopo=tTopoHand.product();
   
   bool printRtS = true;
   
   //NOTE: This assumes that produce will not be called until the edm::Event used in the previous call
   // has been deleted. This is true for now. In the future, we may have to have the resources own
   // the memory.

   // Tracker hit association  
   trackertruth_.reset( new TrackerHitAssociator(iEvent, config_));
   // CSC hit association
   csctruth_.reset(new CSCHitAssociator(iEvent,iSetup,config_));;
   // DT hit association
   printRtS = false;
   dttruth_.reset( new DTHitAssociator(iEvent,iSetup,config_,printRtS) );  
   // RPC hit association
   rpctruth_.reset( new RPCHitAssociator(iEvent,iSetup,config_) );
   
   MuonAssociatorByHitsHelper::Resources resources = {tTopo, trackertruth_.get(), csctruth_.get(), dttruth_.get(), rpctruth_.get()};

   std::unique_ptr<reco::MuonToTrackingParticleAssociatorBaseImpl> impl{ 
     new MuonToTrackingParticleAssociatorByHitsImpl(hitExtractor_,resources, &helper_) }; 
   std::unique_ptr<reco::MuonToTrackingParticleAssociator> toPut( new reco::MuonToTrackingParticleAssociator(std::move(impl)));
   iEvent.put(std::move(toPut));
}

 
// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
MuonToTrackingParticleAssociatorEDProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(MuonToTrackingParticleAssociatorEDProducer);
