// Framework headers
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "IOMC/RandomEngine/interface/TRandomAdaptor.h"

// SimpleConfigurable replacement
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// HepMC headers
#include "HepMC/GenEvent.h"

// Hector headers
#include "SimTransport/HectorProducer/interface/Hector.h"
#include "SimTransport/HectorProducer/interface/HectorProducer.h"

// SimDataFormats headers
#include "SimDataFormats/Forward/interface/LHCTransportLinkContainer.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include <CLHEP/Random/RandomEngine.h>

#include <iostream>
#include <memory>
#include <string>

class TRandom3;

HectorProducer::HectorProducer(edm::ParameterSet const &p)
    : m_HepMC(consumes<edm::HepMCProduct>(p.getParameter<edm::InputTag>("HepMCProductLabel"))) {
  tok_pdt_ = esConsumes<HepPDT::ParticleDataTable, PDTRecord>();
  m_verbosity = p.getParameter<bool>("Verbosity");
  m_FP420Transport = p.getParameter<bool>("FP420Transport");
  m_ZDCTransport = p.getParameter<bool>("ZDCTransport");
  m_evtAnalysed = 0;

  produces<edm::HepMCProduct>();
  produces<edm::LHCTransportLinkContainer>();

  usesResource("Hector");
  m_Hector = std::make_unique<Hector>(p, tok_pdt_, m_verbosity, m_FP420Transport, m_ZDCTransport);

  edm::Service<edm::RandomNumberGenerator> rng;
  if (!rng.isAvailable()) {
    throw cms::Exception("Configuration") << "LHCTransport (HectorProducer) requires the "
                                             "RandomNumberGeneratorService\n"
                                             "which is not present in the configuration file.  You must add the "
                                             "service\n"
                                             "in the configuration file or remove the modules that require it.";
  }
  edm::LogVerbatim("SimTransportHectorProducer") << "Hector is created";
}

HectorProducer::~HectorProducer() {}

void HectorProducer::produce(edm::Event &iEvent, const edm::EventSetup &es) {
  edm::Service<edm::RandomNumberGenerator> rng;
  CLHEP::HepRandomEngine *engine = &rng->getEngine(iEvent.streamID());
  if (engine->name() != "TRandom3") {
    throw cms::Exception("Configuration") << "The TRandom3 engine type must be used with HectorProducer, "
                                          << "Random Number Generator Service not correctly configured!";
  }
  TRandom3 *rootEngine = ((edm::TRandomAdaptor *)engine)->getRootEngine();

  ++m_evtAnalysed;

  edm::LogVerbatim("SimTransportHectorProducer") << "produce evt " << m_evtAnalysed;

  edm::Handle<edm::HepMCProduct> HepMCEvt;
  iEvent.getByToken(m_HepMC, HepMCEvt);

  if (!HepMCEvt.isValid()) {
    throw cms::Exception("InvalidReference") << "Invalid reference to HepMCProduct\n";
  }

  if (HepMCEvt.provenance()->moduleLabel() == "LHCTransport") {
    throw cms::Exception("LogicError") << "HectorTrasported HepMCProduce already exists\n";
  }

  auto evt = new HepMC::GenEvent(*HepMCEvt->GetEvent());
  m_Hector->clearApertureFlags();
  if (m_FP420Transport) {
    m_Hector->clear();
    m_Hector->add(evt, es);
    m_Hector->filterFP420(rootEngine);
  }
  if (m_ZDCTransport) {
    m_Hector->clear();
    m_Hector->add(evt, es);
    m_Hector->filterZDC(rootEngine);

    m_Hector->clear();
    m_Hector->add(evt, es);
    m_Hector->filterD1(rootEngine);
  }
  evt = m_Hector->addPartToHepMC(evt);
  if (m_verbosity)
    evt->print();

  edm::LogVerbatim("SimTransportHectorProducer") << "new HepMC product ";

  unique_ptr<edm::HepMCProduct> NewProduct(new edm::HepMCProduct());
  NewProduct->addHepMCData(evt);

  iEvent.put(std::move(NewProduct));

  edm::LogVerbatim("SimTransportHectorProducer") << "new LHCTransportLinkContainer ";
  unique_ptr<edm::LHCTransportLinkContainer> NewCorrespondenceMap(new edm::LHCTransportLinkContainer());
  edm::LHCTransportLinkContainer thisLink(m_Hector->getCorrespondenceMap());
  (*NewCorrespondenceMap).swap(thisLink);

  if (m_verbosity) {
    for (unsigned int i = 0; i < (*NewCorrespondenceMap).size(); ++i)
      edm::LogVerbatim("HectorEventProcessing") << "Hector correspondence table: " << (*NewCorrespondenceMap)[i];
  }

  iEvent.put(std::move(NewCorrespondenceMap));
  edm::LogVerbatim("SimTransportHectorProducer") << "produce end ";
}

DEFINE_FWK_MODULE(HectorProducer);
