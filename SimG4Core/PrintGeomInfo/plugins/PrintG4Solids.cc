#include "SimG4Core/Notification/interface/BeginOfRun.h"
#include "SimG4Core/Notification/interface/Observer.h"
#include "SimG4Core/Watcher/interface/SimWatcher.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Math/interface/angle_units.h"

#include "G4Box.hh"
#include "G4Cons.hh"
#include "G4Trap.hh"
#include "G4Trd.hh"
#include "G4Tubs.hh"
#include "G4LogicalVolumeStore.hh"
#include "G4LogicalVolume.hh"
#include "G4VSolid.hh"
#include "G4NavigationHistory.hh"
#include "G4PhysicalVolumeStore.hh"
#include "G4TransportationManager.hh"
#include "G4VPhysicalVolume.hh"

#include <algorithm>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>

using angle_units::operators::convertRadToDeg;

class PrintG4Solids : public SimWatcher, public Observer<const BeginOfRun *> {
public:
  PrintG4Solids(edm::ParameterSet const &p);
  ~PrintG4Solids() override = default;

private:
  void update(const BeginOfRun *run) override;
  void dumpSummary(std::ostream &out = G4cout);
  G4VPhysicalVolume *getTopPV();

private:
  G4VPhysicalVolume *theTopPV_;
};

PrintG4Solids::PrintG4Solids(const edm::ParameterSet &p) {
  G4cout << "PrintG4Solids:: initialised for printing information about G4VSolids" << G4endl;
}

void PrintG4Solids::update(const BeginOfRun *run) {
  //Now take action
  theTopPV_ = getTopPV();

  dumpSummary(G4cout);
}

void PrintG4Solids::dumpSummary(std::ostream &out) {
  //---------- Dump number of objects of each class
  out << " @@@@@@@@@@@@@@@@@@ Dumping G4 geometry objects Summary " << G4endl;
  if (theTopPV_ == nullptr) {
    out << " No volume created " << G4endl;
    return;
  }
  out << " @@@ Geometry built inside world volume: " << theTopPV_->GetName() << G4endl;
  // Get number of solids (< # LV if several LV share a solid)
  const G4LogicalVolumeStore *lvs = G4LogicalVolumeStore::GetInstance();
  std::vector<G4LogicalVolume *>::const_iterator lvcite;
  std::set<G4VSolid *> theSolids;
  for (lvcite = lvs->begin(); lvcite != lvs->end(); lvcite++)
    theSolids.insert((*lvcite)->GetSolid());
  out << " Number of G4VSolid's: " << theSolids.size() << G4endl;
  std::set<G4VSolid *>::const_iterator solid;
  for (solid = theSolids.begin(); solid != theSolids.end(); solid++) {
    G4String type = (*solid)->GetEntityType();
    out << (*solid)->GetName() << ":" << type << " Volume " << (*solid)->GetCubicVolume();
    if (type == "G4Box") {
      const G4Box *box = static_cast<const G4Box *>(*solid);
      out << " dx:dy:dz " << box->GetXHalfLength() << ":" << box->GetYHalfLength() << ":" << box->GetZHalfLength();
    } else if (type == "G4Tubs") {
      const G4Tubs *tube = static_cast<const G4Tubs *>(*solid);
      out << " rin:rout:dz:phistart:dphi " << tube->GetInnerRadius() << ":" << tube->GetOuterRadius() << ":"
          << tube->GetZHalfLength() << ":" << convertRadToDeg(tube->GetStartPhiAngle()) << ":"
          << convertRadToDeg(tube->GetDeltaPhiAngle());
    } else if (type == "G4Cons") {
      const G4Cons *cone = static_cast<const G4Cons *>(*solid);
      out << " rinminus:routminus:rinplus:routplus:dz:phistart:dphi " << cone->GetInnerRadiusMinusZ() << ":"
          << cone->GetOuterRadiusMinusZ() << ":" << cone->GetInnerRadiusPlusZ() << ":" << cone->GetOuterRadiusPlusZ()
          << ":" << cone->GetZHalfLength() << ":" << convertRadToDeg(cone->GetStartPhiAngle()) << ":"
          << convertRadToDeg(cone->GetDeltaPhiAngle());
    } else if (type == "G4Trap") {
      const G4Trap *trap = static_cast<const G4Trap *>(*solid);
      out << "zhalf:yl1:xl11:xl12:tana1:yl2:xl21:xl22:tana2 " << trap->GetZHalfLength() << ":"
          << trap->GetYHalfLength1() << ":" << trap->GetXHalfLength1() << ":" << trap->GetXHalfLength2() << ":"
          << trap->GetTanAlpha1() << ":" << trap->GetYHalfLength2() << ":" << trap->GetXHalfLength3() << ":"
          << trap->GetXHalfLength4() << ":" << trap->GetTanAlpha2();
    } else if (type == "G4Trd") {
      const G4Trd *trd = static_cast<const G4Trd *>(*solid);
      out << "xl1:xl2:yl1:yl2:zhalf " << trd->GetXHalfLength1() << ":" << trd->GetXHalfLength2() << ":" << trd->GetYHalfLength1() << ":" << trd->GetYHalfLength2() << ":" << trd->GetZHalfLength();
    }
    out << G4endl;
  }
}

G4VPhysicalVolume *PrintG4Solids::getTopPV() {
  return G4TransportationManager::GetTransportationManager()->GetNavigatorForTracking()->GetWorldVolume();
}

#include "SimG4Core/Watcher/interface/SimWatcherFactory.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"

DEFINE_SIMWATCHER(PrintG4Solids);
