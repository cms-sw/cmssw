#include <iostream>
#include <memory>
#include <thread>
#include <mutex>

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Run.h"

#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "HepPDT/ParticleDataTable.hh"
#include "SimGeneral/HepPDTRecord/interface/PDTRecord.h"

#include "SimG4Core/Application/interface/ExceptionHandler.h"
#include "SimG4Core/Notification/interface/SimG4Exception.h"
#include "SimG4Core/Notification/interface/G4SimEvent.h"
#include "SimG4Core/Geometry/interface/SensitiveDetectorCatalog.h"
#include "SimG4Core/Geometry/interface/DDDWorld.h"

#include "SimG4Core/MagneticField/interface/CMSFieldManager.h"
#include "SimG4Core/MagneticField/interface/Field.h"
#include "SimG4Core/MagneticField/interface/FieldBuilder.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/DDCMS/interface/DDCompactView.h"

#include "G4VPhysicalVolume.hh"
#include "G4PhysicalVolumeStore.hh"
#include "G4RunManagerKernel.hh"
#include "G4TransportationManager.hh"
#include "G4StateManager.hh"

class GeometryMTProducer : public edm::stream::EDProducer< > {
public:

  explicit GeometryMTProducer(edm::ParameterSet const& p);
  ~GeometryMTProducer() override;

  void beginRun(const edm::Run& r, const edm::EventSetup& es) override;
  void endRun(const edm::Run& r, const edm::EventSetup& es) override;
  void produce(edm::Event& e, const edm::EventSetup& es) override;

private:

  void InitialiseTokens(edm::ConsumesCollector&& iC);

  G4VPhysicalVolume* m_world = nullptr;

  // ES products needed for Geant4 initialization
  edm::ParameterSet m_parField;
  edm::ESGetToken<cms::DDCompactView, IdealGeometryRecord> m_DD4Hep; 
  edm::ESGetToken<DDCompactView, IdealGeometryRecord> m_DDD;
  edm::ESGetToken<HepPDT::ParticleDataTable, PDTRecord> m_PDT; 
  edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> m_MagField; 

  mutable const DDCompactView* m_pDDD = nullptr;
  mutable const cms::DDCompactView* m_pDD4Hep = nullptr;
  mutable const HepPDT::ParticleDataTable* m_pTable = nullptr;

  G4RunManagerKernel* m_kernel;
  const MagneticField* m_pMagField = nullptr;

  mutable std::mutex m_threadMutex;

  bool m_geoFromDD4hep;
  bool m_useMagneticField;
};

namespace edm {
  class EventSetup;
}

namespace sim {
  class FieldBuilder;
}

namespace cms {
  class DDCompactView;
}

namespace HepPDT {
  class ParticleDataTable;
}

GeometryMTProducer::GeometryMTProducer(edm::ParameterSet const& p)
  : m_parField(p.getParameter<edm::ParameterSet>("MagneticField")),
    m_geoFromDD4hep(p.getParameter<bool>("g4GeometryDD4hepSource")),
    m_useMagneticField(p.getParameter<bool>("UseMagneticField"))
{
  edm::LogVerbatim("SimG4CoreApplication") << "GeometryMTProducer is constructed";
  m_kernel = new G4RunManagerKernel();
  InitialiseTokens(consumesCollector());
  produces<int>();
}

GeometryMTProducer::~GeometryMTProducer() {
  delete m_kernel;
}

void GeometryMTProducer::beginRun(const edm::Run&, const edm::EventSetup& es) {
  edm::LogVerbatim("SimG4CoreApplication") << "GeometryMTProducer::beginRun";

    
  G4String name = m_geoFromDD4hep ? "OCMS_1" : "OCMS";
  m_world = G4PhysicalVolumeStore::GetInstance()->GetVolume(name, false);
  edm::LogVerbatim("SimG4CoreApplication") << "World Volume " << name << " is accessed: " << (nullptr != m_world);

  // unique initialisation in one thread only
  if(nullptr == m_world) {
    // Lock the mutex
    std::unique_lock<std::mutex> lk(m_threadMutex);
    if(nullptr == m_world) {

      if (m_geoFromDD4hep) {
	m_pDD4Hep = &(*es.getTransientHandle(m_DD4Hep));
      } else {
	m_pDDD = &(*es.getTransientHandle(m_DDD));
      }
      m_pTable = &es.getData(m_PDT);

      SensitiveDetectorCatalog catalog;
      const DDDWorld *dddworld = new DDDWorld(m_pDDD, m_pDD4Hep, catalog, 1, false, false);
      m_world = dddworld->GetWorldVolume();
      if (nullptr != m_world)
	edm::LogVerbatim("SimG4CoreApplication") << "World Volume is built: " << m_world->GetName();
    }
    lk.unlock();
  }

  // world volume in each thread
  m_kernel->DefineWorldVolume(m_world, true);
  G4StateManager::GetStateManager()->SetNewState(G4State_GeomClosed);

  // Geant4 exceptions
  double th = 0.5*CLHEP::GeV;
  G4StateManager::GetStateManager()->SetExceptionHandler(new ExceptionHandler(th));

  // Geant4 transport
  G4TransportationManager* tM = G4TransportationManager::GetTransportationManager();
  tM->SetWorldForTracking(m_world);

  // magnetic field
  if (m_useMagneticField) {
    m_pMagField = &es.getData(m_MagField);
    const GlobalPoint g(0.f, 0.f, 0.f);

    sim::FieldBuilder fieldBuilder(m_pMagField, m_parField);

    CMSFieldManager* fieldManager = new CMSFieldManager();
    tM->SetFieldManager(fieldManager);
    fieldBuilder.build(fieldManager, tM->GetPropagatorInField());
  }

  G4StateManager::GetStateManager()->SetNewState(G4State_Idle);
  edm::LogVerbatim("SimG4CoreApplication")
      << "GeometryMTProducer::beginRun done " 
      << " DD4Hep: " << m_geoFromDD4hep 
      << "; MagField: " << m_useMagneticField; 
}

void GeometryMTProducer::InitialiseTokens(edm::ConsumesCollector&& iC) {
  if (m_geoFromDD4hep) {
    m_DD4Hep = iC.esConsumes<cms::DDCompactView, IdealGeometryRecord, edm::Transition::BeginRun>();
  } else {
    m_DDD = iC.esConsumes<DDCompactView, IdealGeometryRecord, edm::Transition::BeginRun>();
  }
  m_PDT = iC.esConsumes<HepPDT::ParticleDataTable, PDTRecord, edm::Transition::BeginRun>();
  if (m_useMagneticField) {
    m_MagField = iC.esConsumes<MagneticField, IdealMagneticFieldRecord, edm::Transition::BeginRun>();
  }
}

void GeometryMTProducer::endRun(const edm::Run&, const edm::EventSetup&) {
  edm::LogVerbatim("SimG4CoreApplication") << "GeometryMTProducer::endRun";
}

void GeometryMTProducer::produce(edm::Event&, const edm::EventSetup&) {
}

DEFINE_FWK_MODULE(GeometryMTProducer);
