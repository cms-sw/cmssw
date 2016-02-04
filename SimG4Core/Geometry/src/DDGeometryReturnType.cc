#include "SimG4Core/Geometry/interface/DDGeometryReturnType.h"

DDGeometryReturnType::DDGeometryReturnType(G4LogicalVolume* log,
					   G4LogicalVolumeToDDLogicalPartMap map,
					   SensitiveDetectorCatalog catalog) :
  log_(log), map_(map), catalog_(catalog) {}

DDGeometryReturnType:: ~DDGeometryReturnType() {}
