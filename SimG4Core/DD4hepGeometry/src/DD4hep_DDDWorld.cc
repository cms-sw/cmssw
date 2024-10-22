#include "SimG4Core/DD4hepGeometry/interface/DD4hep_DDDWorld.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetectorDescription/DDCMS/interface/DDDetector.h"
#include "DDG4/Geant4Converter.h"
#include "DDG4/Geant4GeometryInfo.h"
#include "DDG4/Geant4Mapping.h"
#include "DD4hep/Detector.h"
#include "DD4hep/Printout.h"

#include "G4RunManagerKernel.hh"
#include "G4PVPlacement.hh"
#include "G4TransportationManager.hh"

using namespace edm;
using namespace cms;
using namespace dd4hep;
using namespace dd4hep::sim;

DDDWorld::DDDWorld(const DDDetector* ddd, dd4hep::sim::Geant4GeometryMaps::VolumeMap& map) {
  LogVerbatim("SimG4CoreApplication") << "DD4hep_DDDWorld: initialization of DDDWorld...";

  DetElement world = ddd->description()->world();
  printout(INFO, "SimDD4CMS", "+++ DDDWorld::DDDWorld start... %s", world.name());
  const Detector& detector = *ddd->description();
  Geant4Converter g4Geo(detector);
  Geant4GeometryInfo* geometry = g4Geo.create(world).detach();
  map = geometry->g4Volumes;

  auto it = geometry->g4Volumes.find(detector.worldVolume().ptr());
  LogVerbatim("Geometry") << "The world is " << it->first.name();

  if (geometry) {
    LogVerbatim("Geometry").log([&](auto& log) {
      for (auto iter = map.begin(); iter != map.end(); ++iter) {
        log << iter->first.name() << " = ";
        if (iter->second)
          log << iter->second->GetName() << "; ";
        else
          log << "***none***; ";
      }
      log << "\n";
    });
  }
  m_world = geometry->world();

  setAsWorld(m_world);
  printout(INFO, "SimDD4CMS", "+++ DDDWorld::DDDWorld done!");

  LogVerbatim("SimG4CoreApplication") << "DD4hep_DDDWorld: initialization of DDDWorld done.";
}

DDDWorld::~DDDWorld() {}

void DDDWorld::setAsWorld(G4VPhysicalVolume* pv) {
  G4RunManagerKernel* kernel = G4RunManagerKernel::GetRunManagerKernel();

  if (kernel)
    kernel->DefineWorldVolume(pv);
  else
    edm::LogError("SimG4CoreGeometry") << "cms::DDDWorld::setAsWorld: No G4RunManagerKernel?";

  edm::LogInfo("SimG4CoreGeometry") << " World volume defined ";
}

void DDDWorld::workerSetAsWorld(G4VPhysicalVolume* pv) {
  G4RunManagerKernel* kernel = G4RunManagerKernel::GetRunManagerKernel();
  if (kernel) {
    kernel->WorkerDefineWorldVolume(pv);
    // The following does not get done in WorkerDefineWorldVolume()
    // because we don't use G4MTRunManager
    G4TransportationManager* transM = G4TransportationManager::GetTransportationManager();
    transM->SetWorldForTracking(pv);
  } else
    edm::LogError("SimG4CoreGeometry") << "cms::DDDWorld::workerSetAsWorld: No G4RunManagerKernel?";

  edm::LogInfo("SimG4CoreGeometry") << " World volume defined (for worker) ";
}
