// Framework headers
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "IOMC/RandomEngine/src/TRandomAdaptor.h"

// SimpleConfigurable replacement
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// HepMC headers
#include "HepMC/GenEvent.h"

// Hector headers
#include "SimTransport/HectorProducer/interface/HectorProducer.h"
#include "SimTransport/HectorProducer/interface/Hector.h"

// SimDataFormats headers
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/Forward/interface/LHCTransportLinkContainer.h"

#include "CLHEP/Random/RandomEngine.h"

#include <iostream>
#include <memory>
#include <string>

class TRandom3;

using std::cout;
using std::endl;

HectorProducer::HectorProducer(edm::ParameterSet const & parameters): eventsAnalysed(0) {
  
  
  // TransportHector
  
  m_InTag          = parameters.getParameter<std::string>("HepMCProductLabel") ;
  m_verbosity      = parameters.getParameter<bool>("Verbosity");
  m_FP420Transport = parameters.getParameter<bool>("FP420Transport");
  m_ZDCTransport   = parameters.getParameter<bool>("ZDCTransport");
  
  produces<edm::HepMCProduct>();
  produces<edm::LHCTransportLinkContainer>();

  hector = new Hector(parameters, 
		      m_verbosity,
		      m_FP420Transport,
		      m_ZDCTransport);
  
  edm::Service<edm::RandomNumberGenerator> rng;
  if ( ! rng.isAvailable() ) {
    throw cms::Exception("Configuration")
      << "LHCTransport (HectorProducer) requires the RandomNumberGeneratorService\n"
         "which is not present in the configuration file.  You must add the service\n"
         "in the configuration file or remove the modules that require it.";
  }
}

HectorProducer::~HectorProducer(){
  
  if(m_verbosity) {
    LogDebug("HectorSetup") << "Delete HectorProducer"  
                            << "Number of events analysed: " << eventsAnalysed;
  }

}

void HectorProducer::produce(edm::Event & iEvent, const edm::EventSetup & es){

  using namespace edm;
  using namespace std;

  edm::Service<edm::RandomNumberGenerator> rng;
  CLHEP::HepRandomEngine* engine = &rng->getEngine(iEvent.streamID());
  if ( engine->name() != "TRandom3" ) {
    throw cms::Exception("Configuration")
      << "The TRandom3 engine type must be used with HectorProducer, Random Number Generator Service not correctly configured!";
  }
  TRandom3* rootEngine = ( (edm::TRandomAdaptor*) engine )->getRootEngine();

  eventsAnalysed++;
  
  Handle<HepMCProduct>  HepMCEvt;   
  iEvent.getByLabel( m_InTag, HepMCEvt ) ;
  
  if ( !HepMCEvt.isValid() )
    {
      throw cms::Exception("InvalidReference")
        << "Invalid reference to HepMCProduct\n";
    }
  
  if ( HepMCEvt.provenance()->moduleLabel() == "LHCTransport" )
    {
      throw cms::Exception("LogicError")
        << "HectorTrasported HepMCProduce already exists\n";
    }

  evt_ = new HepMC::GenEvent( *HepMCEvt->GetEvent() );
  hector->clearApertureFlags();
  if(m_FP420Transport) {
    hector->clear();
    hector->add( evt_ ,es);
    hector->filterFP420(rootEngine);
  }
  if(m_ZDCTransport) {
    hector->clear();
    hector->add( evt_ ,es);
    hector->filterZDC(rootEngine);
    
    hector->clear();
    hector->add( evt_ ,es);
    hector->filterD1(rootEngine);
  }
  evt_ = hector->addPartToHepMC( evt_ );
  if (m_verbosity) {
    evt_->print();
  }
  
  auto_ptr<HepMCProduct> NewProduct(new HepMCProduct()) ;
  NewProduct->addHepMCData( evt_ ) ;
  
  iEvent.put( NewProduct ) ;

  auto_ptr<LHCTransportLinkContainer> NewCorrespondenceMap(new edm::LHCTransportLinkContainer() );
  edm::LHCTransportLinkContainer thisLink(hector->getCorrespondenceMap());
  (*NewCorrespondenceMap).swap(thisLink);

  if ( m_verbosity ) {
    for ( unsigned int i = 0; i < (*NewCorrespondenceMap).size(); i++) 
      LogDebug("HectorEventProcessing") << "Hector correspondence table: " << (*NewCorrespondenceMap)[i];
  }

  iEvent.put( NewCorrespondenceMap );

}

