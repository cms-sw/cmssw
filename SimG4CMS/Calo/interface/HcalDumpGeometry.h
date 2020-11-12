#ifndef SimG4CMS_Calo_HcalDumpGeometry_H
#define SimG4CMS_Calo_HcalDumpGeometry_H
#include "Geometry/HcalCommonData/interface/HcalNumberingFromDDD.h"
#include "SimG4CMS/Calo/interface/HcalTestNumberingScheme.h"
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

class HcalDumpGeometry {
public:
  explicit HcalDumpGeometry(const std::vector<std::string_view> &, const HcalNumberingFromDDD *, bool);
  ~HcalDumpGeometry() = default;

  void update();

private:
  void dumpTouch(G4VPhysicalVolume *pv, unsigned int leafDepth);
  std::string getNameNoNS(const std::string &name);

  const HcalNumberingFromDDD *numberingFromDDD_;
  std::unique_ptr<HcalNumberingScheme> numberingScheme_;
  std::vector<std::string> names_;
  G4NavigationHistory fHistory_;
  std::vector<CaloDetInfo> infoVec_;
};
#endif
