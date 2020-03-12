#ifndef SimG4Core_DDG4Builder_h
#define SimG4Core_DDG4Builder_h

#include "SimG4Core/Geometry/interface/DDG4DispContainer.h"
#include "SimG4Core/Geometry/interface/G4LogicalVolumeToDDLogicalPartMap.h"

#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDSolid.h"

#include <map>
#include <string>
#include <vector>

class DDG4SolidConverter;
class G4LogicalVolume;
class G4VPhysicalVolume;
class G4Material;
class G4VSolid;
class DDCompactView;
class SensitiveDetectorCatalog;

class DDG4Builder {
public:
  DDG4Builder(const DDCompactView *, G4LogicalVolumeToDDLogicalPartMap &, bool check);
  ~DDG4Builder();
  G4LogicalVolume *BuildGeometry(SensitiveDetectorCatalog &);

private:
  G4VSolid *convertSolid(const DDSolid &dSolid);
  G4LogicalVolume *convertLV(const DDLogicalPart &dLogical);
  G4Material *convertMaterial(const DDMaterial &dMaterial);
  int getInt(const std::string &s, const DDLogicalPart &dLogical);
  double getDouble(const std::string &s, const DDLogicalPart &dLogical);

  DDG4SolidConverter *solidConverter_;
  std::map<DDMaterial, G4Material *> mats_;
  std::map<DDSolid, G4VSolid *> sols_;
  std::map<DDLogicalPart, G4LogicalVolume *> logs_;

  const DDCompactView *compactView_;
  G4LogicalVolumeToDDLogicalPartMap &map_;
  DDG4DispContainer *theVectorOfDDG4Dispatchables_;
  bool check_;
};

#endif
