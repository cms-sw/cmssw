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

  hector = new Hector(parameters, 
		      m_verbosity,
		      m_FP420Transport,
		      m_ZDCTransport);
  
  edm::LogInfo ("Hector") << "HectorProducer parameters: \n" 
			  << "   Verbosity: " << m_verbosity << "\n"
			  << "   m_InTag:    " <<  m_InTag<< "\n"
			  << "   m_FP420Transport:    " << m_FP420Transport << "\n"
			  << "   m_ZDCTransport:    " << m_ZDCTransport << "\n";

  if(m_verbosity) {
    cout << "===================================================================" << endl;  
    cout << "=== Start create new HectorProducer                           =====" << endl;
    cout << "=== m_InTag: " << m_InTag << endl;
    cout << "=== You are going to transport:                               =====" << endl;
    cout << "=== FP420: " << m_FP420Transport << endl;
    cout << "=== ZDC: " << m_ZDCTransport << endl;
    cout << "===================================================================" << endl;
  }  
}

HectorProducer::~HectorProducer(){
  
  if(m_verbosity) {
    cout << "===================================================================" << endl;  
    cout << "=== Start delete HectorProducer                               =====" << endl;
    cout << "=== Number of events analysed: " << eventsAnalysed << endl;
  }

  if(m_verbosity) {
    cout << "=== DONE                              =====" << endl;
    cout << "===================================================================" << endl;  
  }
  
}

void HectorProducer::beginJob(const edm::EventSetup & es)
{
}

void HectorProducer::endJob()
{
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
    cout << "HECTOR transported event: " << endl;
    evt_->print();
  }
  
  auto_ptr<HepMCProduct> NewProduct(new HepMCProduct()) ;
  NewProduct->addHepMCData( evt_ ) ;
  
  iEvent.put( NewProduct ) ;
}

