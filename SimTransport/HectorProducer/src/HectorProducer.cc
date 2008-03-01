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
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"

#include <iostream>
#include <memory>

using std::cout;
using std::endl;

HectorProducer::HectorProducer(edm::ParameterSet const & parameters): eventsAnalysed(0) {
  
  
  //  produces<edm::HepMCProduct>();
  
  // TransportHector
  
  m_InTag          = parameters.getParameter<std::string>("HepMCProductLabel") ;
  m_verbosity      = parameters.getParameter<bool>("Verbosity");
  m_FP420Transport = parameters.getParameter<bool>("FP420Transport");
  m_ZDCTransport   = parameters.getParameter<bool>("ZDCTransport");
  
  produces<edm::HepMCProduct>();
  //  hector = new Hector( parameters );
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
  
  //if ( hector ) delete hector;
  
  if(m_verbosity) {
    cout << "===================================================================" << endl;  
    cout << "=== Start delete HectorProducer                               =====" << endl;
    cout << "=== Number of events analysed: " << eventsAnalysed << endl;
  }
  //  delete hector;

   if(m_verbosity) {
    cout << "=== DONE                              =====" << endl;
    cout << "===================================================================" << endl;  
  }
 
}

void HectorProducer::beginJob(const edm::EventSetup & es)
{
  //  cout << "HectorProducer::beginJob" << std::endl;
  //  cout << "" << std::endl;
}

void HectorProducer::endJob()
{
  //    std::cout << " HectorProducer terminating " << std::endl;
}


void HectorProducer::produce(edm::Event & iEvent, const edm::EventSetup & es){
  //  cout << "HectorProducer::produce" << std::endl;
  using namespace edm;
  using namespace std;
  //   using namespace HepMC;
  //   using namespace CLHEP;
  
  eventsAnalysed++;
  
  //   vector< Handle<HepMCProduct> > AllHepMCEvt ;   
  //   iEvent.getManyByType( AllHepMCEvt ) ;
  Handle<HepMCProduct>  HepMCEvt;   
  iEvent.getByLabel( m_InTag, HepMCEvt ) ;
  
  //   for ( unsigned int i=0; i<HepMCEvt.size(); ++i )
  //   {
  if ( !HepMCEvt.isValid() )
    {
      // in principal, should never happen, as it's taken care of bt Framework
      throw cms::Exception("InvalidReference")
	<< "Invalid reference to HepMCProduct\n";
    }
  
  if ( HepMCEvt.provenance()->moduleLabel() == "HectorTrasported" )
    {
      throw cms::Exception("LogicError")
	<< "HectorTrasported HepMCProduce already exists\n";
    }
  
  //   }
  
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
  
  // OK, create a product and put in into edm::Event
  //
  auto_ptr<HepMCProduct> NewProduct(new HepMCProduct()) ;
  NewProduct->addHepMCData( evt_ ) ;
  
  //   iEvent.put( NewProduct, "HectorTrasported" ) ;
  iEvent.put( NewProduct ) ;
}

