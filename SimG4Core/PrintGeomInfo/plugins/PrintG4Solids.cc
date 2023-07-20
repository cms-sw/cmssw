#include "SimG4Core/Notification/interface/BeginOfRun.h"
#include "SimG4Core/Notification/interface/Observer.h"
#include "SimG4Core/Watcher/interface/SimWatcher.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Math/interface/angle_units.h"

#include "G4Box.hh"
#include "G4Cons.hh"
#include "G4ExtrudedSolid.hh"
#include "G4Polycone.hh"
#include "G4Polyhedra.hh"
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
#include <map>
#include <set>
#include <sstream>
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
  bool select(const std::string& name, const std::string& shape) const;
  std::string reducedName(const std::string& name);

private:
  const bool dd4hep_;
  const std::vector<std::string> solids_;
  const std::vector<std::string> types_;
  G4VPhysicalVolume *theTopPV_;
};

PrintG4Solids::PrintG4Solids(const edm::ParameterSet &p) : dd4hep_(p.getUntrackedParameter<bool>("dd4hep")), solids_(p.getUntrackedParameter<std::vector<std::string> >("dumpVolumes")), types_(p.getUntrackedParameter<std::vector<std::string> >("dumpShapes")) {
  G4cout << "PrintG4Solids:: initialised for printing information about G4VSolids for version dd4heP:" << dd4hep_ << G4endl;
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
  for (lvcite = lvs->begin(); lvcite != lvs->end(); lvcite++) {
    G4VSolid* solid = (*lvcite)->GetSolid();
    std::string name = static_cast<std::string>(solid->GetName());
    if (dd4hep_)
      name = reducedName(name);
    std::string type = static_cast<std::string>(solid->GetEntityType());
    if (select(name, type))
      theSolids.insert(solid);
  }
  out << " Number of G4VSolid's: " << theSolids.size() << G4endl;
  std::set<G4VSolid *>::const_iterator solid;
  for (solid = theSolids.begin(); solid != theSolids.end(); solid++) {
    G4String type = (*solid)->GetEntityType();
    std::string name = static_cast<std::string>((*solid)->GetName());
    if (dd4hep_)
      name = reducedName(name);
    out << name << ":" << type << " Volume " << (*solid)->GetCubicVolume();
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
      out << " zhalf:yl1:xl11:xl12:tana1:yl2:xl21:xl22:tana2 " << trap->GetZHalfLength() << ":"
          << trap->GetYHalfLength1() << ":" << trap->GetXHalfLength1() << ":" << trap->GetXHalfLength2() << ":"
          << trap->GetTanAlpha1() << ":" << trap->GetYHalfLength2() << ":" << trap->GetXHalfLength3() << ":"
          << trap->GetXHalfLength4() << ":" << trap->GetTanAlpha2();
    } else if (type == "G4Trd") {
      const G4Trd *trd = static_cast<const G4Trd *>(*solid);
      out << " xl1:xl2:yl1:yl2:zhalf " << trd->GetXHalfLength1() << ":" << trd->GetXHalfLength2() << ":"
          << trd->GetYHalfLength1() << ":" << trd->GetYHalfLength2() << ":" << trd->GetZHalfLength();
    } else if (type == "G4Polycone") {
      const G4Polycone *cone = static_cast<const G4Polycone *>(*solid);
      const auto hist = cone->GetOriginalParameters();
      int num = hist->Num_z_planes;
      out << " angle " << convertRadToDeg(hist->Start_angle) << ":" << convertRadToDeg(hist->Opening_angle) << " with "
          << num << " planes:";
      for (int k = 0; k < num; ++k)
        out << " [" << k << "] " << hist->Z_values[k] << ":" << hist->Rmin[k] << ":" << hist->Rmax[k];
    } else if (type == "G4Polyhedra") {
      const G4Polyhedra *pgon = static_cast<const G4Polyhedra *>(*solid);
      const auto hist = pgon->GetOriginalParameters();
      int num = hist->Num_z_planes;
      out << " angle " << convertRadToDeg(hist->Start_angle) << ":" << convertRadToDeg(hist->Opening_angle) << " with "
          << hist->numSide << " sides and " << num << " planes:";
      for (int k = 0; k < num; ++k)
        out << " [" << k << "] " << hist->Z_values[k] << ":" << hist->Rmin[k] << ":" << hist->Rmax[k];
    } else if (type == "G4ExtrudedSolid") {
      const G4ExtrudedSolid *pgon = static_cast<const G4ExtrudedSolid *>(*solid);
      int vert = pgon->GetNofVertices();
      int numz = pgon->GetNofZSections();
      out << " " << vert << " vertices:";
      for (int k = 0; k < vert; ++k)
        out << " [" << k << "] " << pgon->GetVertex(k);
      out << "; and " << numz << " z-sections:";
      for (int k = 0; k < numz; ++k) {
        const auto &zsec = pgon->GetZSection(k);
        out << " [" << k << "] " << zsec.fZ << ":" << zsec.fScale << ":" << zsec.fOffset;
      }
    }
    out << G4endl;
  }
}
    
G4VPhysicalVolume *PrintG4Solids::getTopPV() {
  return G4TransportationManager::GetTransportationManager()->GetNavigatorForTracking()->GetWorldVolume();
}

std::string PrintG4Solids::reducedName(const std::string& name) {
  std::string nam(name);
  uint32_t first = ((name.find(":") == std::string::npos) ? 0 : (name.find(":") 
								 + 1));
  uint32_t last(name.size() + 1);
  uint32_t loc(first);
  while (1) {
    if (name.find("_", loc) == std::string::npos)
      break;
    if (((loc + 5) < name.size()) && (name.substr(loc, 5) == "shape")) {
      last = loc;
      break;
    }
    loc = name.find("_", loc) + 1;
    if (loc > name.size())
      break;
  }
  nam = name.substr(first, last - first - 1);
  if ((last < name.size()) && (name.substr(name.size() - 5, 5) == "_refl"))
    nam += "_refl";
  return nam;
}

bool PrintG4Solids::select(const std::string& name, const std::string& type) const {
  bool flag(true);
  if (!solids_.empty()) 
    flag = (flag && (std::find(solids_.begin(), solids_.end(), name) != solids_.end()));
  if (!types_.empty())
    flag = (flag && (std::find(types_.begin(), types_.end(), type) != types_.end()));
  return flag;
}

#include "SimG4Core/Watcher/interface/SimWatcherFactory.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"

DEFINE_SIMWATCHER(PrintG4Solids);
