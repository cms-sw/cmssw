#ifndef SimG4Core_DDGeometryReturnType_h
#define SimG4Core_DDGeometryReturnType_h

#include "SimG4Core/Geometry/interface/G4LogicalVolumeToDDLogicalPartMap.h"
#include "SimG4Core/Geometry/interface/SensitiveDetectorCatalog.h"

#include "G4LogicalVolume.hh"

class DDGeometryReturnType {
public:
  DDGeometryReturnType(G4LogicalVolume *log,
                       const G4LogicalVolumeToDDLogicalPartMap &map,
                       const SensitiveDetectorCatalog &catalog);
  ~DDGeometryReturnType();

  G4LogicalVolume *logicalVolume() const { return log_; }
  G4LogicalVolumeToDDLogicalPartMap lvToDDLPMap() const { return map_; }
  SensitiveDetectorCatalog sdCatalog() const { return catalog_; }

private:
  G4LogicalVolume *log_;
  G4LogicalVolumeToDDLogicalPartMap map_;
  SensitiveDetectorCatalog catalog_;
};

#endif
