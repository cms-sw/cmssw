// Framework headers
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
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

#include <iostream>
#include <memory>

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
    hector->filterFP420();
  }
  if(m_ZDCTransport) {
    hector->clear();
    hector->add( evt_ ,es);
    hector->filterZDC();
    
    hector->clear();
    hector->add( evt_ ,es);
    hector->filterD1();
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

