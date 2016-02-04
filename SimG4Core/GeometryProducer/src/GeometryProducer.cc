#include "FWCore/PluginManager/interface/PluginManager.h"

#include "SimG4Core/GeometryProducer/interface/GeometryProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "SimG4Core/Watcher/interface/SimProducer.h"
#include "SimG4Core/Watcher/interface/SimWatcherFactory.h"
#include "SimG4Core/Geometry/interface/DDDWorld.h"
#include "SimG4Core/Geometry/interface/G4LogicalVolumeToDDLogicalPartMap.h"
#include "SimG4Core/Geometry/interface/SensitiveDetectorCatalog.h"
#include "SimG4Core/MagneticField/interface/FieldBuilder.h"
#include "SimG4Core/MagneticField/interface/Field.h"
#include "SimG4Core/Notification/interface/SimG4Exception.h"
#include "SimG4Core/Application/interface/SimTrackManager.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "DetectorDescription/Core/interface/DDCompactView.h"

#include "G4RunManagerKernel.hh"
#include "G4TransportationManager.hh"

#include <iostream>

static
void createWatchers(const edm::ParameterSet& iP, SimActivityRegistry& iReg,
		    std::vector<std::shared_ptr<SimWatcher> >& oWatchers,
		    std::vector<std::shared_ptr<SimProducer> >& oProds)
{
    using namespace std;
    using namespace edm;
    std::vector<ParameterSet> watchers;
    try { watchers = iP.getParameter<vector<ParameterSet> >("Watchers"); } 
    catch(edm::Exception) {}
  
    for(std::vector<ParameterSet>::iterator itWatcher = watchers.begin();
	itWatcher != watchers.end(); ++itWatcher) 
    {
	std::auto_ptr<SimWatcherMakerBase> 
	    maker(SimWatcherFactory::get()->create(itWatcher->getParameter<std::string> ("type")));
	if(maker.get()==0) { throw SimG4Exception("Unable to find the requested Watcher"); }
    
	std::shared_ptr<SimWatcher> watcherTemp;
	std::shared_ptr<SimProducer> producerTemp;
	maker->make(*itWatcher,iReg,watcherTemp,producerTemp);
	oWatchers.push_back(watcherTemp);
	if(producerTemp) oProds.push_back(producerTemp);
    }
}

GeometryProducer::GeometryProducer(edm::ParameterSet const & p) :
    m_kernel(nullptr), 
    m_pUseMagneticField(p.getParameter<bool>("UseMagneticField")),
    m_pField(p.getParameter<edm::ParameterSet>("MagneticField")), 
    m_pUseSensitiveDetectors(p.getParameter<bool>("UseSensitiveDetectors")),
    m_attach(nullptr), m_p(p), m_firstRun ( true )
{
    //Look for an outside SimActivityRegistry
    //this is used by the visualization code
    edm::Service<SimActivityRegistry> otherRegistry;
    if (otherRegistry) m_registry.connect(*otherRegistry);
    createWatchers(m_p, m_registry, m_watchers, m_producers);
    produces<int>();
}

GeometryProducer::~GeometryProducer() 
{ 
  delete m_attach;
  delete m_kernel; 
}

void GeometryProducer::updateMagneticField( edm::EventSetup const& es) {
    if (m_pUseMagneticField)
    {
       // setup the magnetic field
       edm::ESHandle<MagneticField> pMF;
       es.get<IdealMagneticFieldRecord>().get(pMF);
       const GlobalPoint g(0.,0.,0.);
       edm::LogInfo("GeometryProducer") << "B-field(T) at (0,0,0)(cm): " << pMF->inTesla(g);

       m_fieldBuilder = std::auto_ptr<sim::FieldBuilder>(new sim::FieldBuilder(&(*pMF), m_pField));

           G4TransportationManager * tM = G4TransportationManager::GetTransportationManager();
        // update field here ...          
        m_fieldBuilder->build( tM->GetFieldManager(),tM->GetPropagatorInField());

        edm::LogInfo("GeometryProducer") << "Magentic field updated";
    }

}
 
void GeometryProducer::beginLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const& es) {
       // mag field can change in new lumi section
       updateMagneticField( es );
}

void GeometryProducer::beginRun(const edm::Run &, const edm::EventSetup&)
{}

void GeometryProducer::endRun(const edm::Run &,const edm::EventSetup&)
{}

void GeometryProducer::produce(edm::Event & e, const edm::EventSetup & es)
{
    if ( !m_firstRun ) return;
    m_firstRun = false;

    edm::LogInfo("GeometryProducer") << "Producing G4 Geom";   

    m_kernel = G4RunManagerKernel::GetRunManagerKernel();   
    if (m_kernel==0) m_kernel = new G4RunManagerKernel();
    edm::LogInfo("GeometryProducer") << " GeometryProducer initializing ";
    // DDDWorld: get the DDCV from the ES and use it to build the World
    edm::ESTransientHandle<DDCompactView> pDD;
    es.get<IdealGeometryRecord>().get(pDD);
   
    G4LogicalVolumeToDDLogicalPartMap map_;
    SensitiveDetectorCatalog catalog_;
    const DDDWorld * world = new DDDWorld(&(*pDD), map_, catalog_, false);
    m_registry.dddWorldSignal_(world);

    updateMagneticField( es );

    if (m_pUseSensitiveDetectors)
    {
       edm::LogInfo("GeometryProducer") << " instantiating sensitive detectors ";
       // instantiate and attach the sensitive detectors
       m_trackManager = std::auto_ptr<SimTrackManager>(new SimTrackManager);
       if (m_attach==0) m_attach = new AttachSD;
       {
           std::pair< std::vector<SensitiveTkDetector*>,
               std::vector<SensitiveCaloDetector*> > 
             sensDets = m_attach->create(*world,(*pDD),catalog_,m_p,m_trackManager.get(),m_registry);
          
           m_sensTkDets.swap(sensDets.first);
           m_sensCaloDets.swap(sensDets.second);
       }

       edm::LogInfo("GeometryProducer") << " Sensitive Detector building finished; found " 
					<< m_sensTkDets.size()
					<< " Tk type Producers, and " << m_sensCaloDets.size() 
					<< " Calo type producers ";
    }

    for(Producers::iterator itProd = m_producers.begin();itProd != m_producers.end();
           ++itProd) { 
        (*itProd)->produce(e,es); 
    }
}

DEFINE_FWK_MODULE(GeometryProducer);
