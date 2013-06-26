#include "SimG4Core/Geometry/interface/DDDWorld.h"
#include "SimG4Core/Geometry/interface/DDG4Builder.h"

#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4RunManagerKernel.hh"
#include "G4PVPlacement.hh"
 
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
  if (kernel != 0) kernel->DefineWorldVolume(pv);
  edm::LogInfo("SimG4CoreGeometry") << " World volume defined ";
}

