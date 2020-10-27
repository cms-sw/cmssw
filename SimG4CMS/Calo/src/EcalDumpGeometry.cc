#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/EcalCommonData/interface/EcalBarrelNumberingScheme.h"
#include "Geometry/EcalCommonData/interface/EcalBaseNumber.h"
#include "Geometry/EcalCommonData/interface/EcalEndcapNumberingScheme.h"
#include "Geometry/EcalCommonData/interface/EcalPreshowerNumberingScheme.h"
#include "SimG4CMS/Calo/interface/EcalDumpGeometry.h"

#include <iostream>

//#define EDM_ML_DEBUG

EcalDumpGeometry::EcalDumpGeometry(const std::vector<std::string_view>& names, int type) : type_(type) {
  std::stringstream ss;
  for (const auto& lvname : names)
    ss << " " << lvname;
  edm::LogVerbatim("EcalGeom") << " Type: " << type << " with " << names.size() << " LVs: " << ss.str();
  for (const auto& name : names) {
    std::string namex = (static_cast<std::string>(name)).substr(0, 4);
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

  const std::string& lvname = lv->GetName();
  EcalBaseNumber theBaseNumber;
  for (unsigned int k = 0; k < names_.size(); ++k) {
    if (lvname.substr(0, 4) == names_[k]) {
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
          theBaseNumber.addLevel(fHistory_.GetVolume(ii)->GetName(), fHistory_.GetVolume(ii)->GetCopyNo());
#ifdef EDM_ML_DEBUG
          ss << " " << ii << " " << fHistory_.GetVolume(ii)->GetName() << ":" << fHistory_.GetVolume(ii)->GetCopyNo();
#endif
        }
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("EcalGeom") << " Fielde: " << ss.str();
#endif
        uint32_t id = ((type_ == 0) ? ebNumbering_.getUnitID(theBaseNumber)
                                    : ((type_ == 1) ? eeNumbering_.getUnitID(theBaseNumber)
                                                    : esNumbering_.getUnitID(theBaseNumber)));
        std::vector<double> pars;
        if (type_ > 1) {
          G4Box* solid = static_cast<G4Box*>(lv->GetSolid());
          pars.emplace_back(solid->GetXHalfLength());
          pars.emplace_back(solid->GetYHalfLength());
          pars.emplace_back(solid->GetZHalfLength());
        } else {
          G4Trap* solid = static_cast<G4Trap*>(lv->GetSolid());
          pars.emplace_back(solid->GetZHalfLength());
          pars.emplace_back(solid->GetYHalfLength1());
          pars.emplace_back(solid->GetXHalfLength1());
          pars.emplace_back(solid->GetXHalfLength2());
          double a1 = (std::abs(solid->GetTanAlpha1()) > 1.e-5) ? solid->GetTanAlpha1() : 0.0;
          pars.emplace_back(a1);
          pars.emplace_back(solid->GetYHalfLength2());
          pars.emplace_back(solid->GetXHalfLength3());
          pars.emplace_back(solid->GetXHalfLength4());
          double a2 = (std::abs(solid->GetTanAlpha2()) > 1.e-5) ? solid->GetTanAlpha2() : 0.0;
          pars.emplace_back(a2);
        }
        infoVec_.emplace_back(CaloDetInfo(id, lvname, globalpoint, pars));
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
