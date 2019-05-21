#include "SimG4Core/DD4hepGeometry/interface/DD4hep_DDDWorld.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetectorDescription/DDCMS/interface/DDDetector.h"
#include "DDG4/Geant4Converter.h"
#include "DDG4/Geant4GeometryInfo.h"
#include "DD4hep/Detector.h"

#include "G4RunManagerKernel.hh"
#include "G4PVPlacement.hh"
#include "G4TransportationManager.hh"
 
using namespace edm;
using namespace cms;

DDDWorld::DDDWorld(const DDDetector* ddd) {
  
  dd4hep::DetElement world = ddd->description()->world();
  const dd4hep::Detector& detector = *ddd->description();
  dd4hep::sim::Geant4Converter g4Geo(detector);
  dd4hep::sim::Geant4GeometryInfo* geometry = g4Geo.create(world).detach();
 
  m_world = geometry->world();

  SetAsWorld(m_world);
}

DDDWorld::~DDDWorld() {}

void
DDDWorld::SetAsWorld(G4VPhysicalVolume * pv) {
  G4RunManagerKernel* kernel = G4RunManagerKernel::GetRunManagerKernel();
  if(kernel) kernel->DefineWorldVolume(pv);
  else edm::LogError("SimG4CoreGeometry") << "No G4RunManagerKernel?";
  edm::LogInfo("SimG4CoreGeometry") << " World volume defined ";
}

void
DDDWorld::WorkerSetAsWorld(G4VPhysicalVolume * pv) {
  G4RunManagerKernel* kernel = G4RunManagerKernel::GetRunManagerKernel();
  if(kernel) {
    kernel->WorkerDefineWorldVolume(pv);
    // The following does not get done in WorkerDefineWorldVolume()
    // because we don't use G4MTRunManager
    G4TransportationManager* transM = G4TransportationManager::GetTransportationManager();
    transM->SetWorldForTracking(pv);
  }
  else edm::LogError("SimG4CoreGeometry") << "No G4RunManagerKernel?";
  edm::LogInfo("SimG4CoreGeometry") << " World volume defined (for worker) ";
}

