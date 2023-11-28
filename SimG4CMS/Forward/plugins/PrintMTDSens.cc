//#define EDM_ML_DEBUG

#include "SimG4Core/Notification/interface/BeginOfRun.h"
#include "SimG4Core/Notification/interface/Observer.h"
#include "SimG4Core/Watcher/interface/SimWatcher.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/MTDCommonData/interface/BTLNumberingScheme.h"
#include "Geometry/MTDCommonData/interface/ETLNumberingScheme.h"
#include "DataFormats/ForwardDetId/interface/BTLDetId.h"
#include "DataFormats/ForwardDetId/interface/ETLDetId.h"
#include "DataFormats/Math/interface/Rounding.h"

#include "G4Run.hh"
#include "G4VPhysicalVolume.hh"
#include "G4LogicalVolume.hh"
#include "G4NavigationHistory.hh"
#include "G4TransportationManager.hh"
#include "G4Box.hh"

#include <set>
#include <map>
#include <iostream>
#include <string>
#include <vector>

class PrintMTDSens : public SimWatcher, public Observer<const BeginOfRun *> {
public:
  PrintMTDSens(edm::ParameterSet const &p);
  ~PrintMTDSens() override;

private:
  void update(const BeginOfRun *run) override;
  int dumpTouch(G4VPhysicalVolume *pv, unsigned int leafDepth, int ns);
  G4VPhysicalVolume *getTopPV();
  bool getBaseNumber();

private:
  std::vector<std::string> names_;
  std::vector<size_t> nsens_;
  G4NavigationHistory fHistory;

  MTDBaseNumber thisN_;
  BTLNumberingScheme btlNS_;
  ETLNumberingScheme etlNS_;
};

PrintMTDSens::PrintMTDSens(const edm::ParameterSet &p) : thisN_(), btlNS_(), etlNS_() {
  names_ = p.getUntrackedParameter<std::vector<std::string>>("Name");
  nsens_.resize(names_.size());
  for (size_t index = 0; index < nsens_.size(); index++) {
    nsens_[index] = 0;
  }
  G4cout << "PrintMTDSens:: Print position of MTD Sensitive Touchables: " << G4endl;
  for (const auto &thisName : names_) {
    G4cout << " for name " << thisName << "\n";
  }
  G4cout << " Total of " << names_.size() << " sensitive volume types" << G4endl;
}

PrintMTDSens::~PrintMTDSens() {}

void PrintMTDSens::update(const BeginOfRun *run) {
  G4VPhysicalVolume *theTopPV = getTopPV();
  int ntotal = dumpTouch(theTopPV, 0, 0);
  G4cout << "\nTotal number of sensitive detector volumes for MTD is " << ntotal << G4endl;
  for (size_t index = 0; index < nsens_.size(); index++) {
    G4cout << "Sensitive volume " << names_[index] << " # copies " << nsens_[index] << G4endl;
  }
}

using cms_rounding::roundIfNear0;

int PrintMTDSens::dumpTouch(G4VPhysicalVolume *pv, unsigned int leafDepth, int ns) {
  if (leafDepth == 0)
    fHistory.SetFirstEntry(pv);
  else
    fHistory.NewLevel(pv, kNormal, pv->GetCopyNo());

  int ntotal(ns);
  G4ThreeVector globalpoint = fHistory.GetTopTransform().Inverse().TransformPoint(G4ThreeVector(0, 0, 0));
  G4LogicalVolume *lv = pv->GetLogicalVolume();

  std::string mother = "World";
  bool printIt(false);
  if (pv->GetMotherLogical()) {
    mother = pv->GetMotherLogical()->GetName();
  }
  size_t index(0);
  for (const auto &thisName : names_) {
    G4String g4name(thisName);
    if (G4StrUtil::contains(lv->GetName(), g4name)) {
      printIt = true;
      break;
    }
    index++;
  }

  if (lv->GetSensitiveDetector() && printIt) {
    std::stringstream sunitt;
    ++ntotal;
    ++nsens_[index];
    bool isBarrel = getBaseNumber();
    if (isBarrel) {
      BTLDetId theId(btlNS_.getUnitID(thisN_));
      sunitt << theId.rawId();
#ifdef EDM_ML_DEBUG
      G4cout << theId << G4endl;
#endif
    } else {
      ETLDetId theId(etlNS_.getUnitID(thisN_));
      sunitt << theId.rawId();
#ifdef EDM_ML_DEBUG
      G4cout << theId << G4endl;
#endif
    }

    auto fround = [&](double in) {
      std::stringstream ss;
      ss << std::fixed << std::setw(14) << roundIfNear0(in);
      return ss.str();
    };

    G4Box *thisSens = static_cast<G4Box *>(lv->GetSolid());
    G4ThreeVector cn1Global = fHistory.GetTopTransform().Inverse().TransformPoint(
        G4ThreeVector(thisSens->GetXHalfLength(), thisSens->GetYHalfLength(), thisSens->GetZHalfLength()));

    sunitt << fround(globalpoint.x()) << fround(globalpoint.y()) << fround(globalpoint.z()) << fround(cn1Global.x())
           << fround(cn1Global.y()) << fround(cn1Global.z());
    edm::LogVerbatim("MTDG4sensUnitTest") << sunitt.str();
  }

  int NoDaughters = lv->GetNoDaughters();
  while ((NoDaughters--) > 0) {
    G4VPhysicalVolume *pvD = lv->GetDaughter(NoDaughters);
    if (!pvD->IsReplicated())
      ntotal = dumpTouch(pvD, leafDepth + 1, ntotal);
  }

  if (leafDepth > 0)
    fHistory.BackLevel();
  return ntotal;
}

G4VPhysicalVolume *PrintMTDSens::getTopPV() {
  return G4TransportationManager::GetTransportationManager()->GetNavigatorForTracking()->GetWorldVolume();
}

bool PrintMTDSens::getBaseNumber() {
  bool isBTL(false);
  thisN_.reset();
  thisN_.setSize(fHistory.GetMaxDepth());
  int theSize = fHistory.GetDepth() + 1;
  if (thisN_.getCapacity() < theSize)
    thisN_.setSize(theSize);
  //Get name and copy numbers
  if (theSize > 1) {
#ifdef EDM_ML_DEBUG
    G4cout << "Building MTD basenumber:" << G4endl;
#endif
    for (int ii = theSize; ii-- > 0;) {
      thisN_.addLevel(fHistory.GetVolume(ii)->GetName(), fHistory.GetReplicaNo(ii));
#ifdef EDM_ML_DEBUG
      G4cout << "PrintMTDSens::getBaseNumber(): Adding level " << theSize - 1 - ii << ": "
             << fHistory.GetVolume(ii)->GetName() << "[" << fHistory.GetReplicaNo(ii) << "]" << G4endl;
#endif
      if (!isBTL) {
        isBTL = G4StrUtil::contains(fHistory.GetVolume(ii)->GetName(), "BarrelTimingLayer");
      }
    }
  }
  return isBTL;
}

#include "SimG4Core/Watcher/interface/SimWatcherFactory.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"

DEFINE_SIMWATCHER(PrintMTDSens);
