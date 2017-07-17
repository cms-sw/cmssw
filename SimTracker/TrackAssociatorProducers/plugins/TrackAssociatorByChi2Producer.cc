// -*- C++ -*-
//
// Package:    SimTracker/TrackAssociatorProducers
// Class:      TrackAssociatorByChi2Producer
// 
/**\class TrackAssociatorByChi2Producer TrackAssociatorByChi2Producer.cc SimTracker/TrackAssociatorProducers/plugins/TrackAssociatorByChi2Producer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Christopher Jones
//         Created:  Tue, 06 Jan 2015 16:13:00 GMT
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimDataFormats/Associations/interface/TrackToTrackingParticleAssociator.h"
#include "SimDataFormats/Associations/interface/TrackToGenParticleAssociator.h"
#include "SimDataFormats/Associations/interface/TrackToTrackingParticleAssociatorBaseImpl.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h" 

#include "TrackAssociatorByChi2Impl.h"
#include "TrackGenAssociatorByChi2Impl.h"

//
// class declaration
//

class TrackAssociatorByChi2Producer : public edm::global::EDProducer<> {
public:
  explicit TrackAssociatorByChi2Producer(const edm::ParameterSet&);
  ~TrackAssociatorByChi2Producer();
  
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  
private:
  virtual void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
  
  // ----------member data ---------------------------
  edm::EDGetTokenT<reco::BeamSpot> bsToken_;
  const double chi2cut_;
  const bool onlyDiagonal_;
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
TrackAssociatorByChi2Producer::TrackAssociatorByChi2Producer(const edm::ParameterSet& iConfig):
  bsToken_(consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>("beamSpot"))),
  chi2cut_(iConfig.getParameter<double>("chi2cut")),
  onlyDiagonal_(iConfig.getParameter<bool>("onlyDiagonal"))
{
   //register your products
   produces<reco::TrackToTrackingParticleAssociator>();
   produces<reco::TrackToGenParticleAssociator>();
}


TrackAssociatorByChi2Producer::~TrackAssociatorByChi2Producer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
TrackAssociatorByChi2Producer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const
{
   using namespace edm;

   edm::ESHandle<MagneticField> magField;
   iSetup.get<IdealMagneticFieldRecord>().get(magField);

   edm::Handle<reco::BeamSpot> beamSpot;
   iEvent.getByToken(bsToken_,beamSpot);

   {
     std::unique_ptr<reco::TrackToTrackingParticleAssociatorBaseImpl> impl( new TrackAssociatorByChi2Impl(*magField,
                                                                                                        *beamSpot,
                                                                                                          chi2cut_,
                                                                                                          onlyDiagonal_));
     
     std::unique_ptr<reco::TrackToTrackingParticleAssociator> assoc(new reco::TrackToTrackingParticleAssociator(std::move(impl)));
     
     iEvent.put(std::move(assoc));
   }

   {
     std::unique_ptr<reco::TrackToGenParticleAssociatorBaseImpl> impl( new TrackGenAssociatorByChi2Impl(*magField,
                                                                                                        *beamSpot,
                                                                                                        chi2cut_,
                                                                                                        onlyDiagonal_));
     
     std::unique_ptr<reco::TrackToGenParticleAssociator> assoc(new reco::TrackToGenParticleAssociator(std::move(impl)));
     
     iEvent.put(std::move(assoc));
   }
}

 
// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
TrackAssociatorByChi2Producer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(TrackAssociatorByChi2Producer);
