#ifndef SimG4Core_DDG4ProductionCuts_H
#define SimG4Core_DDG4ProductionCuts_H

#include "SimG4Core/Geometry/interface/G4LogicalVolumeToDDLogicalPartMap.h"
#include <DD4hep/SpecParRegistry.h>
#include <DDG4/Geant4GeometryInfo.h>

#include <string>
#include <vector>

class DDLogicalPart;
class G4Region;
class G4LogicalVolume;

class DDG4ProductionCuts {
public:
  explicit DDG4ProductionCuts(const G4LogicalVolumeToDDLogicalPartMap*, int, bool);

  explicit DDG4ProductionCuts(const dd4hep::SpecParRegistry*,
                              const dd4hep::sim::Geant4GeometryMaps::VolumeMap*,
                              int,
                              bool);

private:
  void initialize();
  void setProdCuts(const DDLogicalPart, G4Region*);

  const G4LogicalVolumeToDDLogicalPartMap* map_ = nullptr;
  G4LogicalVolumeToDDLogicalPartMap::Vector vec_;

  // ---------------------------------
  // DD4hep specific initialization,
  //   methods, and local variables...
  void dd4hepInitialize();
  void setProdCuts(const dd4hep::SpecPar*, G4Region*);

  const dd4hep::sim::Geant4GeometryMaps::VolumeMap* dd4hepMap_ = nullptr;
  std::vector<std::pair<G4LogicalVolume*, const dd4hep::SpecPar*>> dd4hepVec_;
  const dd4hep::SpecParRegistry* specPars_;
  // ... end here.
  // ---------------------------------

  const std::string keywordRegion_;
  const int verbosity_;
  const bool protonCut_;
};

#endif
