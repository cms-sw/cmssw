#ifndef SimG4CMS_Calo_EcalDumpGeometry_H
#define SimG4CMS_Calo_EcalDumpGeometry_H
#include "Geometry/EcalCommonData/interface/EcalBarrelNumberingScheme.h"
#include "Geometry/EcalCommonData/interface/EcalBaseNumber.h"
#include "Geometry/EcalCommonData/interface/EcalEndcapNumberingScheme.h"
#include "Geometry/EcalCommonData/interface/EcalPreshowerNumberingScheme.h"
#include "SimG4CMS/Calo/interface/CaloDetInfo.h"

#include "G4Box.hh"
#include "G4LogicalVolumeStore.hh"
#include "G4PhysicalVolumeStore.hh"
#include "G4Run.hh"
#include "G4TransportationManager.hh"
#include "G4Trap.hh"
#include "G4LogicalVolume.hh"
#include "G4VPhysicalVolume.hh"
#include "G4VSolid.hh"

#include <string>
#include <vector>

class EcalDumpGeometry {
public:
  explicit EcalDumpGeometry(const std::vector<std::string_view> &, int);
  ~EcalDumpGeometry() = default;

  void update();

private:
  void dumpTouch(G4VPhysicalVolume *pv, unsigned int leafDepth);
  std::string noRefl(const std::string &name);

  EcalBarrelNumberingScheme ebNumbering_;
  EcalEndcapNumberingScheme eeNumbering_;
  EcalPreshowerNumberingScheme esNumbering_;
  std::vector<std::string> names_;
  int type_;
  G4NavigationHistory fHistory_;
  std::vector<CaloDetInfo> infoVec_;
};
#endif
