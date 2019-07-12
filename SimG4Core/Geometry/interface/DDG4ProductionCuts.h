#ifndef SimG4Core_DDG4ProductionCuts_H
#define SimG4Core_DDG4ProductionCuts_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimG4Core/Geometry/interface/G4LogicalVolumeToDDLogicalPartMap.h"

#include <string>
#include <vector>

class DDLogicalPart;
class G4Region;
class G4LogicalVolume;
class G4ProductionCuts;

class DDG4ProductionCuts {
public:
  DDG4ProductionCuts(const G4LogicalVolumeToDDLogicalPartMap&, int, const edm::ParameterSet&);
  ~DDG4ProductionCuts();

private:
  void initialize();
  void setProdCuts(const DDLogicalPart, G4Region*);

  const G4LogicalVolumeToDDLogicalPartMap& map_;
  // Legacy flag
  bool protonCut_;
  std::string keywordRegion_;
  int verbosity_;
  G4LogicalVolumeToDDLogicalPartMap::Vector vec_;
};

#endif
