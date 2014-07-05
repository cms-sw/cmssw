#include "SimG4Core/Geometry/interface/DDDWorld.h"
#include "SimG4Core/Geometry/interface/DDG4Builder.h"

#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4RunManagerKernel.hh"
#include "G4PVPlacement.hh"
#include "G4TransportationManager.hh"
 
using namespace edm;

DDDWorld::DDDWorld(const DDCompactView* cpv, 
		   G4LogicalVolumeToDDLogicalPartMap & map,
		   SensitiveDetectorCatalog & catalog,
		   bool check) {

  std::auto_ptr<DDG4Builder> theBuilder(new DDG4Builder(cpv, check));

  DDGeometryReturnType ret = theBuilder->BuildGeometry();
  G4LogicalVolume *    world = ret.logicalVolume();

  m_world = new G4PVPlacement(0,G4ThreeVector(),world,"DDDWorld",0,false,0);
  SetAsWorld(m_world);
  map     = ret.lvToDDLPMap();
  catalog = ret.sdCatalog();
}

DDDWorld::~DDDWorld() {}

void DDDWorld::SetAsWorld(G4VPhysicalVolume * pv) {
  G4RunManagerKernel * kernel = G4RunManagerKernel::GetRunManagerKernel();
  if(kernel) kernel->DefineWorldVolume(pv);
  else edm::LogError("SimG4CoreGeometry") << "No G4RunManagerKernel?";
  edm::LogInfo("SimG4CoreGeometry") << " World volume defined ";
}

void DDDWorld::WorkerSetAsWorld(G4VPhysicalVolume * pv) {
  G4RunManagerKernel * kernel = G4RunManagerKernel::GetRunManagerKernel();
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

