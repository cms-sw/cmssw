#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/EcalCommonData/interface/EcalBarrelNumberingScheme.h"
#include "Geometry/EcalCommonData/interface/EcalBaseNumber.h"
#include "Geometry/EcalCommonData/interface/EcalEndcapNumberingScheme.h"
#include "Geometry/EcalCommonData/interface/EcalPreshowerNumberingScheme.h"
#include "SimG4CMS/Calo/interface/EcalDumpGeometry.h"

#include "DD4hep/Filter.h"

#include <iostream>

//#define EDM_ML_DEBUG

EcalDumpGeometry::EcalDumpGeometry(const std::vector<std::string_view>& names, int type) : type_(type) {
  std::stringstream ss;
  for (const auto& lvname : names)
    ss << " " << lvname;
  edm::LogVerbatim("EcalGeom") << " Type: " << type << " with " << names.size() << " LVs: " << ss.str();
  for (const auto& name : names) {
    std::string namex = (static_cast<std::string>(dd4hep::dd::noNamespace(name))).substr(0, 4);
    if (std::find(names_.begin(), names_.end(), namex) == names_.end())
      names_.emplace_back(namex);
  }
  edm::LogVerbatim("EcalGeom") << "EcalDumpGeometry:: dump geometry information for detector of type " << type_
                               << " with " << names_.size() << " elements:";
  for (unsigned int k = 0; k < names_.size(); ++k)
    edm::LogVerbatim("EcalGeom") << "[" << k << "] : " << names_[k];
}

void EcalDumpGeometry::update() {
  G4VPhysicalVolume* theTopPV =
      G4TransportationManager::GetTransportationManager()->GetNavigatorForTracking()->GetWorldVolume();
  edm::LogVerbatim("EcalGeom") << "EcalDumpGeometry entered with entry of top PV at " << theTopPV;

  dumpTouch(theTopPV, 0);
  fHistory_.SetFirstEntry(theTopPV);
  edm::LogVerbatim("EcalGeom") << "EcalDumpGeometry finds " << infoVec_.size() << " touchables";
  sort(infoVec_.begin(), infoVec_.end(), CaloDetInfoLess());
  unsigned int k(0);
  for (const auto& info : infoVec_) {
    edm::LogVerbatim("EcalGeom") << "[" << k << "] " << info;
    if (info.flag() && (info.solid() != nullptr)) {
      info.solid()->DumpInfo();
      G4cout << G4endl;
    }
    ++k;
  }
}

void EcalDumpGeometry::dumpTouch(G4VPhysicalVolume* pv, unsigned int leafDepth) {
  if (leafDepth == 0)
    fHistory_.SetFirstEntry(pv);
  else
    fHistory_.NewLevel(pv, kNormal, pv->GetCopyNo());

  G4ThreeVector globalpoint = fHistory_.GetTopTransform().Inverse().TransformPoint(G4ThreeVector(0, 0, 0));
  G4LogicalVolume* lv = pv->GetLogicalVolume();

  bool flag = ((type_ / 10) % 10 > 0);
  std::string lvname = (static_cast<std::string>(dd4hep::dd::noNamespace(lv->GetName())));
  std::string namex = lvname.substr(0, 4);
  EcalBaseNumber theBaseNumber;
  for (unsigned int k = 0; k < names_.size(); ++k) {
    if (namex == names_[k]) {
      int theSize = fHistory_.GetDepth();
      //Get name and copy numbers
      if (theSize > 5) {
        theBaseNumber.reset();
        if (theBaseNumber.getCapacity() < theSize + 1)
          theBaseNumber.setSize(theSize + 1);
#ifdef EDM_ML_DEBUG
        std::stringstream ss;
#endif
        for (int ii = theSize; ii >= 0; --ii) {
          std::string_view name = dd4hep::dd::noNamespace(fHistory_.GetVolume(ii)->GetName());
          theBaseNumber.addLevel(static_cast<std::string>(name), fHistory_.GetVolume(ii)->GetCopyNo());
#ifdef EDM_ML_DEBUG
          ss << " " << ii << " " << name << ":" << fHistory_.GetVolume(ii)->GetCopyNo();
#endif
        }
        uint32_t id = (((type_ % 10) == 0) ? ebNumbering_.getUnitID(theBaseNumber)
                                           : (((type_ % 10) == 1) ? eeNumbering_.getUnitID(theBaseNumber)
                                                                  : esNumbering_.getUnitID(theBaseNumber)));
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("EcalGeom") << " Field: " << ss.str() << " ID " << std::hex << id << std::dec;
#endif
        G4VSolid* solid = (lv->GetSolid());
        if (((type_ / 100) % 10) != 0)
          infoVec_.emplace_back(CaloDetInfo(id, noRefl(lvname), globalpoint, solid, flag));
        else
          infoVec_.emplace_back(CaloDetInfo(id, lvname, globalpoint, solid, flag));
      }
      break;
    }
  }

  int NoDaughters = lv->GetNoDaughters();
  while ((NoDaughters--) > 0) {
    G4VPhysicalVolume* pvD = lv->GetDaughter(NoDaughters);
    if (!pvD->IsReplicated())
      dumpTouch(pvD, leafDepth + 1);
  }

  if (leafDepth > 0)
    fHistory_.BackLevel();
}

std::string EcalDumpGeometry::noRefl(const std::string& name) {
  if (name.find("_refl") == std::string::npos) {
    return name;
  } else {
    size_t n = name.size();
    return name.substr(0, n - 5);
  }
}
