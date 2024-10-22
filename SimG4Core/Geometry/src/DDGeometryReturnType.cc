#include "SimG4Core/Geometry/interface/DDGeometryReturnType.h"

DDGeometryReturnType::DDGeometryReturnType(G4LogicalVolume *log,
                                           const G4LogicalVolumeToDDLogicalPartMap &map,
                                           const SensitiveDetectorCatalog &catalog)
    : log_(log), map_(map), catalog_(catalog) {}

DDGeometryReturnType::~DDGeometryReturnType() {}
