#include "SimMuon/CSCDigitizer/src/CSCDigiProducer.h"

#include "DataFormats/Common/interface/Handle.h"

#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "SimMuon/CSCDigitizer/src/CSCConfigurableStripConditions.h"
#include "SimMuon/CSCDigitizer/src/CSCDbStripConditions.h"

#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <FWCore/MessageLogger/interface/MessageLogger.h>

#include <string>

CSCDigiProducer::CSCDigiProducer(const edm::ParameterSet& ps) 
:  theDigitizer(ps),
   theStripConditions(0)
{
  produces<CSCWireDigiCollection>("MuonCSCWireDigi");
  produces<CSCStripDigiCollection>("MuonCSCStripDigi");
  produces<CSCComparatorDigiCollection>("MuonCSCComparatorDigi");
  produces<DigiSimLinks>("MuonCSCWireDigiSimLinks");
  produces<DigiSimLinks>("MuonCSCStripDigiSimLinks");
  std::string stripConditions( ps.getParameter<std::string>("stripConditions") );
  geometryType = ps.getParameter<std::string>("GeometryType");
  edm::ParameterSet stripPSet = ps.getParameter<edm::ParameterSet>("strips");
  if( stripConditions == "Configurable" )
  {
    theStripConditions = new CSCConfigurableStripConditions(stripPSet);
  }
  else if ( stripConditions == "Database" )
  {
    theStripConditions = new CSCDbStripConditions(stripPSet);
  }
  else
  {
    throw cms::Exception("CSCDigiProducer") 
      << "Bad option for strip conditions: "
      << stripConditions;
  }
  theDigitizer.setStripConditions(theStripConditions);

  edm::Service<edm::RandomNumberGenerator> rng;
  if ( ! rng.isAvailable()) {
   throw cms::Exception("Configuration")
     << "CSCDigitizer requires the RandomNumberGeneratorService\n"
        "which is not present in the configuration file.  You must add the service\n"
        "in the configuration file or remove the modules that require it.";
  }

  std::string mix_ = ps.getParameter<std::string>("mixLabel");
  std::string collection_ = ps.getParameter<std::string>("InputCollection");
  cf_token = consumes<CrossingFrame<PSimHit> >( edm::InputTag(mix_, collection_) );
}


CSCDigiProducer::~CSCDigiProducer()
{
  delete theStripConditions;
}


void CSCDigiProducer::produce(edm::Event& ev, const edm::EventSetup& eventSetup) {

  edm::LogVerbatim("CSCDigitizer") << "[CSCDigiProducer::produce] starting event " << 
      ev.id().event() << " of run " << ev.id().run();
  edm::Service<edm::RandomNumberGenerator> rng;
  CLHEP::HepRandomEngine* engine = &rng->getEngine(ev.streamID());

  edm::Handle<CrossingFrame<PSimHit> > cf;
  ev.getByToken(cf_token, cf);

  std::auto_ptr<MixCollection<PSimHit> > 
    hits( new MixCollection<PSimHit>(cf.product()) );

  // Create empty output

  std::auto_ptr<CSCWireDigiCollection> pWireDigis(new CSCWireDigiCollection());
  std::auto_ptr<CSCStripDigiCollection> pStripDigis(new CSCStripDigiCollection());
  std::auto_ptr<CSCComparatorDigiCollection> pComparatorDigis(new CSCComparatorDigiCollection());
  std::auto_ptr<DigiSimLinks> pWireDigiSimLinks(new DigiSimLinks() );
  std::auto_ptr<DigiSimLinks> pStripDigiSimLinks(new DigiSimLinks() );

  //@@ DOES NOTHING IF NO HITS.  Remove this for when there's real neutrons
  if(hits->size() > 0) 
  {
    // find the geometry & conditions for this event
    edm::ESHandle<CSCGeometry> hGeom;
    eventSetup.get<MuonGeometryRecord>().get(geometryType,hGeom);
    const CSCGeometry *pGeom = &*hGeom;

    theDigitizer.setGeometry( pGeom );

    // find the magnetic field
    edm::ESHandle<MagneticField> magfield;
    eventSetup.get<IdealMagneticFieldRecord>().get(magfield);

    theDigitizer.setMagneticField(&*magfield);


    // set the particle table
    edm::ESHandle < ParticleDataTable > pdt;
    eventSetup.getData( pdt );
    theDigitizer.setParticleDataTable(&*pdt);

    theStripConditions->initializeEvent(eventSetup);

    // run the digitizer
    theDigitizer.doAction(*hits, *pWireDigis, *pStripDigis, *pComparatorDigis,
                          *pWireDigiSimLinks, *pStripDigiSimLinks, engine);
  }


  // store them in the event
  ev.put(pWireDigis, "MuonCSCWireDigi");
  ev.put(pStripDigis, "MuonCSCStripDigi");
  ev.put(pComparatorDigis, "MuonCSCComparatorDigi");
  ev.put(pWireDigiSimLinks, "MuonCSCWireDigiSimLinks");
  ev.put(pStripDigiSimLinks, "MuonCSCStripDigiSimLinks");
}

