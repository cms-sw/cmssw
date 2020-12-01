#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "SimG4CMS/Calo/interface/HcalDumpGeometry.h"

#include <iostream>
#include <memory>

//#define EDM_ML_DEBUG

HcalDumpGeometry::HcalDumpGeometry(const std::vector<std::string_view>& names,
                                   const HcalNumberingFromDDD* hcn,
                                   bool test,
                                   bool flag)
    : numberingFromDDD_(hcn), flag_(flag) {
  if (test)
    numberingScheme_.reset(dynamic_cast<HcalNumberingScheme*>(new HcalTestNumberingScheme(false)));
  else
    numberingScheme_ = std::make_unique<HcalNumberingScheme>();
  std::stringstream ss;
  for (const auto& lvname : names)
    ss << " " << lvname;
  edm::LogVerbatim("HCalGeom") << " Testmode: " << test << " with " << names.size() << " LVs: " << ss.str();
  const std::vector<std::string> namg = {"HBS", "HES", "HTS", "HVQ"};
  for (const auto& name : names) {
    std::string namex = (getNameNoNS(static_cast<std::string>(name))).substr(0, 3);
    if (std::find(namg.begin(), namg.end(), namex) != namg.end()) {
      if (std::find(names_.begin(), names_.end(), namex) == names_.end())
        names_.emplace_back(namex);
    }
  }
  edm::LogVerbatim("HCalGeom") << "HcalDumpGeometry:: dump geometry information for Hcal with " << names_.size()
                               << " elements:";
  for (unsigned int k = 0; k < names_.size(); ++k)
    edm::LogVerbatim("HCalGeom") << "[" << k << "] : " << names_[k];
}

void HcalDumpGeometry::update() {
  G4VPhysicalVolume* theTopPV =
      G4TransportationManager::GetTransportationManager()->GetNavigatorForTracking()->GetWorldVolume();
  edm::LogVerbatim("HCalGeom") << "HcalDumpGeometry entered with entry of top PV at " << theTopPV;

  dumpTouch(theTopPV, 0);
  fHistory_.SetFirstEntry(theTopPV);
  edm::LogVerbatim("HCalGeom") << "HcalDumpGeometry finds " << infoVec_.size() << " touchables";
  sort(infoVec_.begin(), infoVec_.end(), CaloDetInfoLess());
  unsigned int k(0);
  for (const auto& info : infoVec_) {
    edm::LogVerbatim("HCalGeom") << "[" << k << "] " << info;
    if (info.flag() && (info.solid() != nullptr)) {
      info.solid()->DumpInfo();
      G4cout << G4endl;
    }
    ++k;
  }
}

void HcalDumpGeometry::dumpTouch(G4VPhysicalVolume* pv, unsigned int leafDepth) {
  if (leafDepth == 0)
    fHistory_.SetFirstEntry(pv);
  else
    fHistory_.NewLevel(pv, kNormal, pv->GetCopyNo());

  G4ThreeVector globalpoint = fHistory_.GetTopTransform().Inverse().TransformPoint(G4ThreeVector(0, 0, 0));
  G4LogicalVolume* lv = pv->GetLogicalVolume();

  const std::string& lvname = lv->GetName();
  std::string namex = (getNameNoNS(lvname)).substr(0, 3);
  for (unsigned int k = 0; k < names_.size(); ++k) {
    if (namex == names_[k]) {
      int theSize = fHistory_.GetDepth();
      //Get name and copy numbers
      if (theSize > 5) {
        int depth = (fHistory_.GetVolume(theSize)->GetCopyNo()) % 10 + 1;
        int lay = (fHistory_.GetVolume(theSize)->GetCopyNo() / 10) % 100 + 1;
        int det = (fHistory_.GetVolume(theSize - 1)->GetCopyNo()) / 1000;
        HcalNumberingFromDDD::HcalID tmp = numberingFromDDD_->unitID(
            det, math::XYZVectorD(globalpoint.x(), globalpoint.y(), globalpoint.z()), depth, lay);
        uint32_t id = numberingScheme_->getUnitID(tmp);
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HCalGeom") << "Det " << det << " Layer " << lay << ":" << depth << " Volume "
                                     << fHistory_.GetVolume(theSize)->GetName() << ":"
                                     << fHistory_.GetVolume(theSize - 1)->GetName() << " ID " << std::hex << id
                                     << std::dec;
#endif

        G4VSolid* solid = lv->GetSolid();
        infoVec_.emplace_back(CaloDetInfo(id, 0, 0, getNameNoNS(lvname), globalpoint, solid, flag_));
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

std::string HcalDumpGeometry::getNameNoNS(const std::string& name) {
  if (name.find(':') == std::string::npos) {
    return name;
  } else {
    auto n1 = name.find(':') + 1;
    return name.substr(n1, (name.size() - n1));
  }
}
