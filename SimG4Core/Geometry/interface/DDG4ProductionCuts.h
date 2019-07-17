#ifndef SimG4Core_DDG4ProductionCuts_H
#define SimG4Core_DDG4ProductionCuts_H

#include "SimG4Core/Geometry/interface/G4LogicalVolumeToDDLogicalPartMap.h"

#include <string>
#include <vector>

class DDLogicalPart;
class G4Region;
class G4LogicalVolume;

class DDG4ProductionCuts {
public:
  explicit DDG4ProductionCuts(const G4LogicalVolumeToDDLogicalPartMap&, int, bool);
  ~DDG4ProductionCuts();

private:
  void initialize();
  void setProdCuts(const DDLogicalPart, G4Region*);

  const G4LogicalVolumeToDDLogicalPartMap& map_;
  G4LogicalVolumeToDDLogicalPartMap::Vector vec_;
  const std::string keywordRegion_;
  const int verbosity_;
  const bool protonCut_;
};

#endif
