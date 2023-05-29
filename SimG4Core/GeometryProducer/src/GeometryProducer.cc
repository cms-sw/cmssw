#include "FWCore/PluginManager/interface/PluginManager.h"

#include "SimG4Core/GeometryProducer/interface/GeometryProducer.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "SimG4Core/Geometry/interface/DDDWorld.h"
#include "SimG4Core/Geometry/interface/G4LogicalVolumeToDDLogicalPartMap.h"
#include "SimG4Core/Geometry/interface/SensitiveDetectorCatalog.h"
#include "SimG4Core/MagneticField/interface/CMSFieldManager.h"
#include "SimG4Core/MagneticField/interface/Field.h"
#include "SimG4Core/MagneticField/interface/FieldBuilder.h"
#include "SimG4Core/Notification/interface/SimTrackManager.h"
#include "SimG4Core/Watcher/interface/SimProducer.h"
#include "SimG4Core/Watcher/interface/SimWatcherFactory.h"
#include "SimG4Core/SensitiveDetector/interface/sensitiveDetectorMakers.h"
#include "SimG4Core/SensitiveDetector/interface/AttachSD.h"

#include "G4RunManagerKernel.hh"
#include "G4TransportationManager.hh"
#include "G4EmParameters.hh"
#include "G4HadronicParameters.hh"

#include <iostream>
#include <memory>

static void createWatchers(const edm::ParameterSet &iP,
                           SimActivityRegistry &iReg,
                           std::vector<std::shared_ptr<SimWatcher>> &oWatchers,
                           std::vector<std::shared_ptr<SimProducer>> &oProds) {
  using namespace std;
  using namespace edm;
  std::vector<ParameterSet> watchers;
  try {
    watchers = iP.getParameter<vector<ParameterSet>>("Watchers");
  } catch (edm::Exception const &) {
  }

  for (std::vector<ParameterSet>::iterator itWatcher = watchers.begin(); itWatcher != watchers.end(); ++itWatcher) {
    std::unique_ptr<SimWatcherMakerBase> maker(
        SimWatcherFactory::get()->create(itWatcher->getParameter<std::string>("type")));
    if (maker.get() == nullptr) {
      throw cms::Exception("SimG4CoreGeometryProducer", " createWatchers: Unable to find the requested Watcher");
    }

    std::shared_ptr<SimWatcher> watcherTemp;
    std::shared_ptr<SimProducer> producerTemp;
    maker->make(*itWatcher, iReg, watcherTemp, producerTemp);
    oWatchers.push_back(watcherTemp);
    if (producerTemp)
      oProds.push_back(producerTemp);
  }
}

GeometryProducer::GeometryProducer(edm::ParameterSet const &p)
    : m_kernel(nullptr),
      m_pField(p.getParameter<edm::ParameterSet>("MagneticField")),
      m_p(p),
      m_pDD(nullptr),
      m_pDD4hep(nullptr),
      m_verbose(0),
      m_firstRun(true),
      m_pUseMagneticField(p.getParameter<bool>("UseMagneticField")),
      m_pUseSensitiveDetectors(p.getParameter<bool>("UseSensitiveDetectors")),
      m_pGeoFromDD4hep(p.getParameter<bool>("GeoFromDD4hep")) {
  // Look for an outside SimActivityRegistry
  // this is used by the visualization code

  edm::Service<SimActivityRegistry> otherRegistry;
  if (otherRegistry)
    m_registry.connect(*otherRegistry);
  createWatchers(m_p, m_registry, m_watchers, m_producers);

  G4EmParameters::Instance()->SetVerbose(m_verbose);
  G4HadronicParameters::Instance()->SetVerboseLevel(m_verbose);

  m_kernel = G4RunManagerKernel::GetRunManagerKernel();
  if (m_kernel == nullptr)
    m_kernel = new G4RunManagerKernel();

  m_kernel->SetVerboseLevel(m_verbose);

  //if (m_pUseSensitiveDetectors)
  //  m_sdMakers = sim::sensitiveDetectorMakers(m_p, consumesCollector(), std::vector<std::string>());

  tokMF_ = esConsumes<MagneticField, IdealMagneticFieldRecord, edm::Transition::BeginRun>();
  if (m_pGeoFromDD4hep) {
    tokDD4hep_ = esConsumes<cms::DDCompactView, IdealGeometryRecord, edm::Transition::BeginRun>();
  } else {
    tokDDD_ = esConsumes<DDCompactView, IdealGeometryRecord, edm::Transition::BeginRun>();
  }
  produces<int>();
}

GeometryProducer::~GeometryProducer() { delete m_kernel; }

void GeometryProducer::updateMagneticField(edm::EventSetup const &es) {
  if (m_pUseMagneticField) {
    // setup the magnetic field
    auto const &pMF = &es.getData(tokMF_);
    const GlobalPoint g(0., 0., 0.);
    edm::LogInfo("GeometryProducer") << "B-field(T) at (0,0,0)(cm): " << pMF->inTesla(g);

    sim::FieldBuilder fieldBuilder(pMF, m_pField);
    CMSFieldManager *fieldManager = new CMSFieldManager();
    G4TransportationManager *tM = G4TransportationManager::GetTransportationManager();
    tM->SetFieldManager(fieldManager);
    fieldBuilder.build(fieldManager, tM->GetPropagatorInField());
    edm::LogInfo("GeometryProducer") << "Magentic field is built";
  }
}

void GeometryProducer::beginLuminosityBlock(edm::LuminosityBlock &, edm::EventSetup const &) {
  // mag field cannot be change in new lumi section - this is commented out
  //     updateMagneticField( es );
}

void GeometryProducer::beginRun(const edm::Run &run, const edm::EventSetup &es) {
  makeGeom(es);
  updateMagneticField(es);
  for (auto &maker : m_sdMakers) {
    maker.second->beginRun(es);
  }
}

void GeometryProducer::endRun(const edm::Run &, const edm::EventSetup &) {}

void GeometryProducer::produce(edm::Event &e, const edm::EventSetup &es) {
  if (!m_firstRun)
    return;
  m_firstRun = false;
  for (Producers::iterator itProd = m_producers.begin(); itProd != m_producers.end(); ++itProd) {
    (*itProd)->produce(e, es);
  }
}

void GeometryProducer::makeGeom(const edm::EventSetup &es) {
  if (!m_firstRun)
    return;

  edm::LogVerbatim("GeometryProducer") << " GeometryProducer initializing ";
  // DDDWorld: get the DDCV from the ES and use it to build the World
  if (m_pGeoFromDD4hep) {
    m_pDD4hep = &es.getData(tokDD4hep_);
  } else {
    m_pDD = &es.getData(tokDDD_);
  }

  SensitiveDetectorCatalog catalog;
  const DDDWorld *dddworld = new DDDWorld(m_pDD, m_pDD4hep, catalog, m_verbose, false, false);
  G4VPhysicalVolume *world = dddworld->GetWorldVolume();
  if (nullptr != world)
    edm::LogVerbatim("GeometryProducer") << " World Volume: " << world->GetName();
  m_kernel->DefineWorldVolume(world, true);

  m_registry.dddWorldSignal_(dddworld);

  edm::LogVerbatim("GeometryProducer") << " Magnetic field initialisation";
  updateMagneticField(es);

  if (m_pUseSensitiveDetectors) {
    edm::LogInfo("GeometryProducer") << " instantiating sensitive detectors ";
    // instantiate and attach the sensitive detectors
    m_trackManager = std::make_unique<SimTrackManager>();
    {
      std::pair<std::vector<SensitiveTkDetector *>, std::vector<SensitiveCaloDetector *>> sensDets =
          sim::attachSD(m_sdMakers, es, catalog, m_p, m_trackManager.get(), m_registry);

      m_sensTkDets.swap(sensDets.first);
      m_sensCaloDets.swap(sensDets.second);
    }

    edm::LogInfo("GeometryProducer") << " Sensitive Detector building finished; found " << m_sensTkDets.size()
                                     << " Tk type Producers, and " << m_sensCaloDets.size() << " Calo type producers ";
  }
}

DEFINE_FWK_MODULE(GeometryProducer);
