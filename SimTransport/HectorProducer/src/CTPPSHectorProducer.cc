// Framework headers
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
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
#include "SimTransport/HectorProducer/interface/CTPPSHectorProducer.h"
#include "SimTransport/HectorProducer/interface/CTPPSHector.h"

// SimDataFormats headers
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/Forward/interface/LHCTransportLinkContainer.h"

#include "CLHEP/Random/RandomEngine.h"

#include <iostream>
#include <memory>
#include <string>

class TRandom3;

CTPPSHectorProducer::CTPPSHectorProducer(edm::ParameterSet const & parameters):eventsAnalysed(0){

    // TransportHector
    m_InTag          = parameters.getParameter<std::string>("HepMCProductLabel") ;
    m_InTagToken     = consumes<edm::HepMCProduct>(m_InTag);
    m_verbosity      = parameters.getParameter<bool>("Verbosity");
    m_CTPPSTransport   = parameters.getParameter<bool>("CTPPSTransport"); 

    produces<edm::HepMCProduct>();
    produces<edm::LHCTransportLinkContainer>();

    hector_ctpps = new CTPPSHector(parameters, m_verbosity,m_CTPPSTransport);

    edm::Service<edm::RandomNumberGenerator> rng;
    if ( ! rng.isAvailable() ) {
        throw cms::Exception("Configuration")
            << "LHCTransport (CTPPSHectorProducer) requires the RandomNumberGeneratorService\n"
            "which is not present in the configuration file.  You must add the service\n"
            "in the configuration file or remove the modules that require it.";
    }
}

CTPPSHectorProducer::~CTPPSHectorProducer(){}


void CTPPSHectorProducer::beginRun(const edm::Run & r,const edm::EventSetup& c) {}

void CTPPSHectorProducer::endRun(const edm::Run & r,const edm::EventSetup& c) {}

void CTPPSHectorProducer::produce(edm::Event & iEvent, const edm::EventSetup & es){

    using namespace edm;
    using namespace std;

    edm::Service<edm::RandomNumberGenerator> rng;
    CLHEP::HepRandomEngine* engine = &rng->getEngine(iEvent.streamID());
    if ( engine->name() != "TRandom3" ) {
        throw cms::Exception("Configuration")
            << "The TRandom3 engine type must be used with CTPPSHectorProducer, Random Number Generator Service not correctly configured!";
    }
    TRandom3* rootEngine = ( (edm::TRandomAdaptor*) engine )->getRootEngine();

    eventsAnalysed++;
    Handle<HepMCProduct>  HepMCEvt;   
    iEvent.getByToken( m_InTagToken, HepMCEvt ) ;

    if ( !HepMCEvt.isValid() ){
        throw cms::Exception("InvalidReference")
            << "Invalid reference to HepMCProduct\n";
    }

    if ( HepMCEvt.provenance()->moduleLabel() == "LHCTransport" ){
        throw cms::Exception("LogicError")
            << "HectorTrasported HepMCProduce already exists\n";
    }

    evt_ = new HepMC::GenEvent( *HepMCEvt->GetEvent() );
    hector_ctpps->clearApertureFlags();
    if(m_CTPPSTransport) {
        hector_ctpps->clear();
        hector_ctpps->add( evt_ ,es ,engine);
        hector_ctpps->filterCTPPS(rootEngine);
    } 

    evt_ = hector_ctpps->addPartToHepMC( evt_ );
    if (m_verbosity) {
        evt_->print();
    }

    unique_ptr<HepMCProduct> NewProduct(new edm::HepMCProduct()) ;
    NewProduct->addHepMCData( evt_ ) ;

    iEvent.put(std::move(NewProduct)) ;

    unique_ptr<LHCTransportLinkContainer> NewCorrespondenceMap(new edm::LHCTransportLinkContainer() );
    edm::LHCTransportLinkContainer thisLink(hector_ctpps->getCorrespondenceMap());
    (*NewCorrespondenceMap).swap(thisLink);

    if ( m_verbosity ) {
        for ( unsigned int i = 0; i < (*NewCorrespondenceMap).size(); i++) 
            LogDebug("HectorEventProcessing") << "Hector correspondence table: " << (*NewCorrespondenceMap)[i];
    }

    iEvent.put(std::move(NewCorrespondenceMap));
    hector_ctpps->clear();

}
DEFINE_FWK_MODULE (CTPPSHectorProducer);
