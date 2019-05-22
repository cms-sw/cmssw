// -*- C++ -*-
//
// Package:    SimPPS/PPSSimTrackProducer
// Class:      PPSSimTrackProducer
//
/**\class PPSSimTrackProducer PPSSimTrackProducer.cc SimPPS/PPSSimTrackProducer/plugins/PPSSimTrackProducer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Luiz Martins Mundim Filho
//         Created:  Sun, 03 Dec 2017 00:25:54 GMT
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "SimDataFormats/Forward/interface/LHCTransportLink.h"
#include "SimDataFormats/Forward/interface/LHCTransportLinkContainer.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "SimTransport/PPSProtonTransport/interface/TotemTransport.h"
#include "SimTransport/PPSProtonTransport/interface/HectorTransport.h"
#include "SimTransport/PPSProtonTransport/interface/ProtonTransport.h"
#include "TRandom3.h"
#include "IOMC/RandomEngine/src/TRandomAdaptor.h"
//
// class declaration
//

class PPSSimTrackProducer : public edm::stream::EDProducer<> {
public:
  explicit PPSSimTrackProducer(const edm::ParameterSet&);
  ~PPSSimTrackProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginStream(edm::StreamID) override;
  void produce(edm::Event&, const edm::EventSetup&) override;
  void endStream() override;

  // ----------member data ---------------------------
  bool m_verbosity;
  ProtonTransport* theTransporter = nullptr;
  edm::InputTag m_InTag;
  edm::EDGetTokenT<edm::HepMCProduct> m_InTagToken;

  std::string m_transportMethod;
  int m_eventsAnalysed;  //!< just to count events that have been analysed
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
PPSSimTrackProducer::PPSSimTrackProducer(const edm::ParameterSet& iConfig) {
  //now do what ever other initialization is needed
  // TransportHector
  m_InTag = iConfig.getParameter<edm::InputTag>("HepMCProductLabel");
  m_InTagToken = consumes<edm::HepMCProduct>(m_InTag);

  m_verbosity = iConfig.getParameter<bool>("Verbosity");
  m_transportMethod = iConfig.getParameter<std::string>("TransportMethod");

  produces<edm::HepMCProduct>();
  produces<edm::LHCTransportLinkContainer>();

  theTransporter = nullptr;

  if (m_transportMethod == "Totem") {
    theTransporter = new TotemTransport(iConfig, m_verbosity);
  } else if (m_transportMethod == "Hector") {
    theTransporter = new HectorTransport(iConfig, m_verbosity);
  } else {
    throw cms::Exception("Configuration")
        << "LHCTransport (ProtonTransport) requires a Method (Hector or Totem) \n"
           "which is not present in the configuration file. You should add one of the method\n"
           "above in the configuration file or remove the module that requires it.";
  }

  edm::Service<edm::RandomNumberGenerator> rng;
  if (!rng.isAvailable()) {
    throw cms::Exception("Configuration")
        << "LHCTransport (ProtonTransport) requires the RandomNumberGeneratorService\n"
           "which is not present in the configuration file.  You must add the service\n"
           "in the configuration file or remove the modules that require it.";
  }
}

PPSSimTrackProducer::~PPSSimTrackProducer() {
  // do anything here that needs to be done at destruction time
  // (e.g. close files, deallocate resources etc.)
  if (theTransporter) {
    delete theTransporter;
    theTransporter = nullptr;
  }
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void PPSSimTrackProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  using namespace std;
  HepMC::GenEvent* evt;
  edm::Service<edm::RandomNumberGenerator> rng;
  CLHEP::HepRandomEngine* engine = &rng->getEngine(iEvent.streamID());
  if (engine->name() != "TRandom3") {
    throw cms::Exception("Configuration") << "The TRandom3 engine type must be used with ProtonTransport, Random "
                                             "Number Generator Service not correctly configured!";
  }

  m_eventsAnalysed++;
  Handle<HepMCProduct> HepMCEvt;
  iEvent.getByToken(m_InTagToken, HepMCEvt);

  if (!HepMCEvt.isValid()) {
    throw cms::Exception("InvalidReference") << "Invalid reference to HepMCProduct\n";
  }

  if (HepMCEvt.provenance()->moduleLabel() == "LHCTransport") {
    throw cms::Exception("LogicError") << "HectorTrasported HepMCProduce already exists\n";
  }

  evt = new HepMC::GenEvent(*HepMCEvt->GetEvent());

  theTransporter->clear();
  theTransporter->process(evt, iSetup, engine);

  if (m_verbosity)
    evt->print();

  unique_ptr<HepMCProduct> newProduct(new edm::HepMCProduct());
  newProduct->addHepMCData(evt);

  iEvent.put(std::move(newProduct));

  unique_ptr<LHCTransportLinkContainer> NewCorrespondenceMap(new edm::LHCTransportLinkContainer());
  edm::LHCTransportLinkContainer thisLink(theTransporter->getCorrespondenceMap());
  (*NewCorrespondenceMap).swap(thisLink);

  if (m_verbosity) {
    for (unsigned int i = 0; i < (*NewCorrespondenceMap).size(); i++)
      LogDebug("HectorEventProcessing") << "Hector correspondence table: " << (*NewCorrespondenceMap)[i];
  }

  iEvent.put(std::move(NewCorrespondenceMap));
  // There is no need to delete the pointer to the event, since it is deleted in HepMCProduct,
  // in fact, it MUST NOT be delete here, as a protection is missing in above package
}
// The methods below are pure virtual, so it needs to be implemented even if not used
//
// ------------ method called once each stream before processing any runs, lumis or events  ------------
void PPSSimTrackProducer::beginStream(edm::StreamID) {}

// ------------ method called once each stream after processing all runs, lumis and events  ------------
void PPSSimTrackProducer::endStream() {}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void PPSSimTrackProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(PPSSimTrackProducer);
