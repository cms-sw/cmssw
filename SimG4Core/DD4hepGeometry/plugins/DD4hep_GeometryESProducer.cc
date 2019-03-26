#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "SimG4Core/Notification/interface/SimActivityRegistry.h"
#include "SimG4Core/SensitiveDetector/interface/AttachSD.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveDetector.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveTkDetector.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveCaloDetector.h"
#include "SimG4Core/Watcher/interface/SimProducer.h"
#include "SimG4Core/Watcher/interface/SimWatcherFactory.h"
#include "SimG4Core/Geometry/interface/DDDWorld.h"
#include "SimG4Core/Geometry/interface/G4LogicalVolumeToDDLogicalPartMap.h"
#include "SimG4Core/Geometry/interface/SensitiveDetectorCatalog.h"
#include "SimG4Core/MagneticField/interface/FieldBuilder.h"
#include "SimG4Core/MagneticField/interface/CMSFieldManager.h"
#include "SimG4Core/MagneticField/interface/Field.h"
#include "SimG4Core/Notification/interface/SimTrackManager.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "DetectorDescription/Core/interface/DDCompactView.h"

#include "G4RunManagerKernel.hh"
#include "G4TransportationManager.hh"

#include <iostream>

#include <memory>

namespace sim { class FieldBuilder; }

class SimWatcher;
class SimProducer;
class DDDWorld;
class G4RunManagerKernel;
class SimTrackManager;

using namespace edm;
using namespace std;
using namespace cms;

static
void createWatchers(const ParameterSet& iP, SimActivityRegistry& iReg,
		    vector<shared_ptr<SimWatcher>>& oWatchers,
		    vector<shared_ptr<SimProducer>>& oProds)
{
  using Exception = cms::Exception;
  vector<ParameterSet> watchers;
  try { watchers = iP.getParameter<vector<ParameterSet> >("Watchers"); } 
  catch(Exception const&) {}
  
  for(vector<ParameterSet>::iterator itWatcher = watchers.begin();
      itWatcher != watchers.end(); ++itWatcher) {
    unique_ptr<SimWatcherMakerBase> 
      maker(SimWatcherFactory::get()->create(itWatcher->getParameter<std::string> ("type")));
    if(maker.get() == nullptr) { 
      throw Exception("SimG4CoreG4GeometryESProducer", 
		      " createWatchers: Unable to find the requested Watcher");
    }
      
    shared_ptr<SimWatcher> watcherTemp;
    shared_ptr<SimProducer> producerTemp;
    maker->make(*itWatcher,iReg,watcherTemp,producerTemp);
    oWatchers.push_back(watcherTemp);
    if(producerTemp) oProds.push_back(producerTemp);
  }
}

namespace cms {
  
class G4GeometryESProducer : public one::EDProducer<one::SharedResources, one::WatchRuns>
{
public:
  using Producers = vector<shared_ptr<SimProducer>>;
 
  explicit G4GeometryESProducer(ParameterSet const&);
  ~G4GeometryESProducer() override;
  void beginRun(const Run&, const EventSetup&) override;
  void endRun(const Run&, const EventSetup&) override;
  void produce(Event&, const EventSetup&) override;
  void beginLuminosityBlock(LuminosityBlock&, EventSetup const&);
  
  vector<shared_ptr<SimProducer>> producers() const {
    return m_producers;
  }
  vector<SensitiveTkDetector*>& sensTkDetectors() { return m_sensTkDets; }
  vector<SensitiveCaloDetector*>& sensCaloDetectors() { return m_sensCaloDets; }
  
private:

  void updateMagneticField(EventSetup const&);
  
  G4RunManagerKernel* m_kernel;
  bool m_pUseMagneticField;
  ParameterSet m_pField;
  bool m_pUseSensitiveDetectors;
  SimActivityRegistry m_registry;
  vector<shared_ptr<SimWatcher>> m_watchers;
  vector<shared_ptr<SimProducer>> m_producers;
  unique_ptr<sim::FieldBuilder> m_fieldBuilder;
  unique_ptr<SimTrackManager> m_trackManager;
  AttachSD * m_attach;
  vector<SensitiveTkDetector*> m_sensTkDets;
  vector<SensitiveCaloDetector*> m_sensCaloDets;
  ParameterSet m_p;
  bool m_firstRun;
};
}

G4GeometryESProducer::G4GeometryESProducer(ParameterSet const& p) :
    m_kernel(nullptr),
    m_pUseMagneticField(p.getParameter<bool>("UseMagneticField")),
    m_pField(p.getParameter<ParameterSet>("MagneticField")), 
    m_pUseSensitiveDetectors(p.getParameter<bool>("UseSensitiveDetectors")),
    m_attach(nullptr), m_p(p), m_firstRun(true)
{
    //Look for an outside SimActivityRegistry
    //this is used by the visualization code
    Service<SimActivityRegistry> otherRegistry;
    if(otherRegistry) m_registry.connect(*otherRegistry);
    createWatchers(m_p, m_registry, m_watchers, m_producers);
    produces<int>();
}

G4GeometryESProducer::~G4GeometryESProducer() 
{
  delete m_attach;
  delete m_kernel; 
}

void G4GeometryESProducer::updateMagneticField(EventSetup const& es) {
  if(m_pUseMagneticField) {
    // setup the magnetic field
    ESHandle<MagneticField> pMF;
    es.get<IdealMagneticFieldRecord>().get(pMF);
    const GlobalPoint g(0.,0.,0.);
    LogInfo("G4GeometryESProducer") << "B-field(T) at (0,0,0)(cm): " << pMF->inTesla(g);

    sim::FieldBuilder fieldBuilder(pMF.product(), m_pField);
    CMSFieldManager* fieldManager = new CMSFieldManager();
    G4TransportationManager * tM = G4TransportationManager::GetTransportationManager();
    tM->SetFieldManager(fieldManager);
    fieldBuilder.build( fieldManager, tM->GetPropagatorInField());
    LogInfo("G4GeometryESProducer") << "Magentic field is built";
  }
}

void G4GeometryESProducer::beginLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&) 
{
  // mag field cannot be change in new lumi section - this is commented out
  //     updateMagneticField( es );
}

void G4GeometryESProducer::beginRun(const edm::Run &, const edm::EventSetup&)
{}

void G4GeometryESProducer::endRun(const edm::Run &,const edm::EventSetup&)
{}

void G4GeometryESProducer::produce(edm::Event & e, const edm::EventSetup & es)
{
  if(!m_firstRun) return;
  m_firstRun = false;
  
  LogInfo("G4GeometryESProducer") << "Producing G4 Geom";   

  m_kernel = G4RunManagerKernel::GetRunManagerKernel();   
  if(m_kernel == nullptr) m_kernel = new G4RunManagerKernel();
  LogInfo("G4GeometryESProducer") << " G4GeometryESProducer initializing ";
  // DDDWorld: get the DDCV from the ES and use it to build the World
  ESTransientHandle<DDCompactView> pDD;
  es.get<IdealGeometryRecord>().get(pDD);
  //  m_detDesc->apply("DD4hep_XMLLoader", 1, &arg);
  G4LogicalVolumeToDDLogicalPartMap map;
  SensitiveDetectorCatalog catalog;
  const DDDWorld * world = new DDDWorld(&(*pDD), map, catalog, false);
  m_registry.dddWorldSignal_(world);
  //FIXME:??? m_registry.watchDDDWorld(world);

  updateMagneticField(es);
  
  if(m_pUseSensitiveDetectors) {
    LogInfo("G4GeometryESProducer") << " instantiating sensitive detectors ";
    // instantiate and attach the sensitive detectors
    m_trackManager = unique_ptr<SimTrackManager>(new SimTrackManager);
    if(m_attach == nullptr) m_attach = new AttachSD;
    {
      pair<vector<SensitiveTkDetector*>,
	   vector<SensitiveCaloDetector*>> 
	sensDets = m_attach->create((*pDD), catalog, m_p, m_trackManager.get(), m_registry);
      
      m_sensTkDets.swap(sensDets.first);
      m_sensCaloDets.swap(sensDets.second);
    }

    LogInfo("G4GeometryESProducer") << " Sensitive Detector building finished; found " 
				<< m_sensTkDets.size()
				<< " Tk type Producers, and " << m_sensCaloDets.size() 
				<< " Calo type producers ";
  }

  for(Producers::iterator itProd = m_producers.begin();itProd != m_producers.end();
      ++itProd) { 
    (*itProd)->produce(e,es); 
  }
}

DEFINE_FWK_MODULE(G4GeometryESProducer);

