#include "SimG4Core/Geometry/interface/DDDWorld.h"
#include "SimG4Core/Geometry/interface/DDG4Builder.h"

#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/DDCMS/interface/DDDetector.h"
#include "DDG4/Geant4Converter.h"
#include "DDG4/Geant4GeometryInfo.h"
#include "DDG4/Geant4Mapping.h"
#include "DD4hep/Detector.h"
#include "DD4hep/Printout.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4PVPlacement.hh"
#include "G4RunManagerKernel.hh"
#include "G4TransportationManager.hh"

using namespace edm;
using namespace dd4hep;
using namespace dd4hep::sim;

DDDWorld::DDDWorld(const DDCompactView *cpv,
                   G4LogicalVolumeToDDLogicalPartMap &map,
                   SensitiveDetectorCatalog &catalog,
                   bool check) {
  std::unique_ptr<DDG4Builder> theBuilder(new DDG4Builder(cpv, check));

  DDGeometryReturnType ret = theBuilder->BuildGeometry();
  G4LogicalVolume *world = ret.logicalVolume();

  m_world = new G4PVPlacement(nullptr, G4ThreeVector(), world, "DDDWorld", nullptr, false, 0);
  SetAsWorld(m_world);
  map = ret.lvToDDLPMap();
  catalog = ret.sdCatalog();
}

DDDWorld::DDDWorld(const cms::DDDetector *ddd, dd4hep::sim::Geant4GeometryMaps::VolumeMap &map) {
  LogVerbatim("SimG4CoreApplication") << "DD4hep_DDDWorld: initialization of DDDWorld...";

  DetElement world = ddd->description()->world();
  const Detector &detector = *ddd->description();
  Geant4Converter g4Geo(detector);
  Geant4GeometryInfo *geometry = g4Geo.create(world).detach();
  map = geometry->g4Volumes;
  m_world = geometry->world();
  SetAsWorld(m_world);

  LogVerbatim("SimG4CoreApplication") << "DD4hep_DDDWorld: initialization of DDDWorld done.";
}

DDDWorld::~DDDWorld() {}

void DDDWorld::SetAsWorld(G4VPhysicalVolume *pv) {
  G4RunManagerKernel *kernel = G4RunManagerKernel::GetRunManagerKernel();
  if (kernel)
    kernel->DefineWorldVolume(pv);
  else
    edm::LogError("SimG4CoreGeometry") << "No G4RunManagerKernel?";
  edm::LogInfo("SimG4CoreGeometry") << " World volume defined ";
}

void DDDWorld::WorkerSetAsWorld(G4VPhysicalVolume *pv) {
  G4RunManagerKernel *kernel = G4RunManagerKernel::GetRunManagerKernel();
  if (kernel) {
    kernel->WorkerDefineWorldVolume(pv);
    // The following does not get done in WorkerDefineWorldVolume()
    // because we don't use G4MTRunManager
    G4TransportationManager *transM = G4TransportationManager::GetTransportationManager();
    transM->SetWorldForTracking(pv);
  } else
    edm::LogError("SimG4CoreGeometry") << "No G4RunManagerKernel?";
  edm::LogInfo("SimG4CoreGeometry") << " World volume defined (for worker) ";
}
