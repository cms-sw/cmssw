#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/EcalCommonData/interface/EcalBarrelNumberingScheme.h"
#include "Geometry/EcalCommonData/interface/EcalBaseNumber.h"
#include "Geometry/EcalCommonData/interface/EcalEndcapNumberingScheme.h"
#include "Geometry/EcalCommonData/interface/EcalPreshowerNumberingScheme.h"
#include "SimG4Core/Geometry/interface/DD4hep2DDDName.h"
#include "SimG4CMS/Calo/interface/EcalDumpGeometry.h"

#include <iostream>

EcalDumpGeometry::EcalDumpGeometry(const std::vector<std::string_view>& names,
                                   const std::string& name1,
                                   const std::string& name2,
                                   int type)
    : name1_(name1), name2_(name2), type_(type) {
  std::stringstream ss;
  for (const auto& lvname : names)
    ss << " " << lvname;
  G4cout << " Type: " << type << " Depth Names " << name1_ << ":" << name2_ << " with " << names.size()
         << " LVs: " << ss.str() << G4endl;
  for (const auto& name : names) {
    std::string namex = DD4hep2DDDName::noNameSpace(static_cast<std::string>(name)).substr(0, 4);
    if (std::find(names_.begin(), names_.end(), namex) == names_.end())
      names_.emplace_back(namex);
  }
  G4cout << "EcalDumpGeometry:: dump geometry information for detector of type " << type_ << " with " << names_.size()
         << " elements:" << G4endl;
  for (unsigned int k = 0; k < names_.size(); ++k)
    G4cout << "[" << k << "] : " << names_[k] << G4endl;
}

void EcalDumpGeometry::update() {
  G4VPhysicalVolume* theTopPV =
      G4TransportationManager::GetTransportationManager()->GetNavigatorForTracking()->GetWorldVolume();
  G4cout << "EcalDumpGeometry entered with entry of top PV at " << theTopPV << G4endl;

  dumpTouch(theTopPV, 0);
  fHistory_.SetFirstEntry(theTopPV);
  G4cout << "EcalDumpGeometry finds " << infoVec_.size() << " touchables" << G4endl;
  sort(infoVec_.begin(), infoVec_.end(), CaloDetInfoLess());
  unsigned int k(0);
  for (const auto& info : infoVec_) {
    G4cout << "[" << k << "] " << info << G4endl;
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
  std::string lvname = DD4hep2DDDName::noNameSpace(static_cast<std::string>(lv->GetName()));
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
        std::stringstream ss;
        for (int ii = theSize; ii >= 0; --ii) {
          std::string name = DD4hep2DDDName::noNameSpace(static_cast<std::string>(fHistory_.GetVolume(ii)->GetName()));
          theBaseNumber.addLevel(name, fHistory_.GetVolume(ii)->GetCopyNo());
          ss << " " << ii << " " << name << ":" << fHistory_.GetVolume(ii)->GetCopyNo();
        }
        uint32_t id = (((type_ % 10) == 0) ? ebNumbering_.getUnitID(theBaseNumber)
                                           : (((type_ % 10) == 1) ? eeNumbering_.getUnitID(theBaseNumber)
                                                                  : esNumbering_.getUnitID(theBaseNumber)));
        uint32_t depth(0);
        if ((!name1_.empty()) && (namex == name1_))
          depth = 1;
        if ((!name2_.empty()) && (namex == name2_))
          depth = 2;
        double r = globalpoint.rho();
        G4cout << " Field: " << ss.str() << " ID " << std::hex << id << std::dec << ":" << depth << ":" << r << G4endl;
        G4VSolid* solid = (lv->GetSolid());
        if (((type_ / 100) % 10) != 0)
          infoVec_.emplace_back(CaloDetInfo(id, depth, r, noRefl(lvname), globalpoint, solid, flag));
        else
          infoVec_.emplace_back(CaloDetInfo(id, depth, r, lvname, globalpoint, solid, flag));
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
