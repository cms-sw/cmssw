#include "SimG4Core/Notification/interface/BeginOfRun.h"
#include "SimG4Core/Notification/interface/Observer.h"
#include "SimG4Core/Watcher/interface/SimWatcher.h"
#include "SimG4Core/Geometry/interface/DD4hep2DDDName.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Math/interface/angle_units.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "G4LogicalVolumeStore.hh"
#include "G4LogicalVolume.hh"
#include "G4VSolid.hh"
#include "G4Material.hh"
#include "G4NavigationHistory.hh"
#include "G4PhysicalVolumeStore.hh"
#include "G4Region.hh"
#include "G4RegionStore.hh"
#include "G4Run.hh"
#include "G4Track.hh"
#include "G4TransportationManager.hh"
#include "G4UserLimits.hh"
#include "G4VisAttributes.hh"
#include "G4VPhysicalVolume.hh"

#include <algorithm>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>

using angle_units::operators::convertRadToDeg;

class PrintG4Touch : public SimWatcher, public Observer<const BeginOfRun *> {
public:
  PrintG4Touch(edm::ParameterSet const &p);
  ~PrintG4Touch() override = default;

private:
  void update(const BeginOfRun *run) override;
  void dumpSummary(std::ostream &out = G4cout);
  int countNoTouchables();
  void add1touchable(G4LogicalVolume *lv, int &nTouch);
  void dumpTouch(G4VPhysicalVolume *pv, unsigned int leafDepth, std::ostream &out = G4cout);
  void getTouch(G4VPhysicalVolume *pv, unsigned int leafDepth, unsigned int copym, std::vector<std::string> &touches);
  G4VPhysicalVolume *getTopPV();
  G4LogicalVolume *getTopLV();

private:
  bool dd4hep_, verbosity_;
  G4VPhysicalVolume *theTopPV_;
  G4NavigationHistory fHistory_;
};

PrintG4Touch::PrintG4Touch(const edm::ParameterSet &p) {
  dd4hep_ = p.getUntrackedParameter<bool>("DD4hep", false);
  verbosity_ = p.getUntrackedParameter<bool>("Verbosity", false);
  G4cout << "PrintG4Touch:: initialised for dd4hep " << dd4hep_ << " with verbosity levels:"
         << verbosity_ << G4endl;
}

void PrintG4Touch::update(const BeginOfRun *run) {
  //Now take action
  theTopPV_ = getTopPV();

  dumpSummary(G4cout);

  std::vector<std::string> touches;
  getTouch(theTopPV_, 0, 1, touches);
  std::sort(touches.begin(), touches.end());
  for (const auto &touch : touches)
    G4cout << touch << G4endl;

  //---------- Dump LV and PV information
  if (verbosity_)
    dumpTouch(theTopPV_, 0, G4cout);
}

void PrintG4Touch::dumpSummary(std::ostream &out) {
  //---------- Dump number of objects of each class
  out << " @@@@@@@@@@@@@@@@@@ Dumping G4 geometry objects Summary " << G4endl;
  if (theTopPV_ == nullptr) {
    out << " No volume created " << G4endl;
    return;
  }
  out << " @@@ Geometry built inside world volume: " << DD4hep2DDDName::namePV(static_cast<std::string>(theTopPV_->GetName()), dd4hep_) << G4endl;
  // Get number of solids (< # LV if several LV share a solid)
  const G4LogicalVolumeStore *lvs = G4LogicalVolumeStore::GetInstance();
  std::vector<G4LogicalVolume *>::const_iterator lvcite;
  std::set<G4VSolid *> theSolids;
  for (lvcite = lvs->begin(); lvcite != lvs->end(); lvcite++)
    theSolids.insert((*lvcite)->GetSolid());
  out << " Number of G4VSolid's: " << theSolids.size() << G4endl;
  out << " Number of G4LogicalVolume's: " << lvs->size() << G4endl;
  const G4PhysicalVolumeStore *pvs = G4PhysicalVolumeStore::GetInstance();
  out << " Number of G4VPhysicalVolume's: " << pvs->size() << G4endl;
  out << " Number of Touchable's: " << countNoTouchables() << G4endl;
  const G4MaterialTable *matTab = G4Material::GetMaterialTable();
  out << " Number of G4Material's: " << matTab->size() << G4endl;
  const G4RegionStore *regs = G4RegionStore::GetInstance();
  out << " Number of G4Region's: " << regs->size() << G4endl;
}

int PrintG4Touch::countNoTouchables() {
  int nTouch = 0;
  G4LogicalVolume *lv = getTopLV();
  add1touchable(lv, nTouch);
  return nTouch;
}

void PrintG4Touch::add1touchable(G4LogicalVolume *lv, int &nTouch) {
  int siz = lv->GetNoDaughters();
  for (int ii = 0; ii < siz; ii++)
    add1touchable(lv->GetDaughter(ii)->GetLogicalVolume(), ++nTouch);
}

void PrintG4Touch::dumpTouch(G4VPhysicalVolume *pv, unsigned int leafDepth, std::ostream &out) {
  if (leafDepth == 0)
    fHistory_.SetFirstEntry(pv);
  else
    fHistory_.NewLevel(pv, kNormal, pv->GetCopyNo());

  G4LogicalVolume *lv = pv->GetLogicalVolume();

  G4ThreeVector globalpoint = fHistory_.GetTopTransform().Inverse().TransformPoint(G4ThreeVector(0, 0, 0));
  std::string mother = (pv->GetMotherLogical()) ? (DD4hep2DDDName::nameSolid(static_cast<std::string>(pv->GetMotherLogical()->GetSolid()->GetName()), dd4hep_)) : "World";
  std::string lvname = DD4hep2DDDName::nameSolid(static_cast<std::string>(lv->GetName()), dd4hep_);
  out << leafDepth << "### VOLUME = " << lvname << " Copy No " << pv->GetCopyNo() << " in " << mother << " global position of centre " << globalpoint << " (r = " << globalpoint.perp() << ", phi = " << convertRadToDeg(globalpoint.phi()) << ")" << G4endl;

  int NoDaughters = lv->GetNoDaughters();
  while ((NoDaughters--) > 0) {
    G4VPhysicalVolume *pvD = lv->GetDaughter(NoDaughters);
    if (!pvD->IsReplicated())
      dumpTouch(pvD, leafDepth + 1, out);
  }

  if (leafDepth > 0)
    fHistory_.BackLevel();
}

void PrintG4Touch::getTouch(G4VPhysicalVolume *pv,
			    unsigned int leafDepth,
			    unsigned int copym,
			    std::vector<std::string> &touches) {
  if (leafDepth == 0)
    fHistory_.SetFirstEntry(pv);
  else
    fHistory_.NewLevel(pv, kNormal, pv->GetCopyNo());

  std::string mother = (pv->GetMotherLogical()) ? (DD4hep2DDDName::nameSolid(static_cast<std::string>(pv->GetMotherLogical()->GetSolid()->GetName()), dd4hep_)) : "World";

  G4LogicalVolume *lv = pv->GetLogicalVolume();
  std::string lvname = DD4hep2DDDName::nameSolid(static_cast<std::string>(lv->GetName()), dd4hep_);
  unsigned int copy = static_cast<unsigned int>(pv->GetCopyNo());

  std::string type = static_cast<std::string>(lv->GetSolid()->GetEntityType());

  std::string name = lvname + ":" + std::to_string(copy) + "_" + mother + ":" + std::to_string(copym) + ":" + type;
  touches.emplace_back(name);

  int NoDaughters = lv->GetNoDaughters();
  while ((NoDaughters--) > 0) {
    G4VPhysicalVolume *pvD = lv->GetDaughter(NoDaughters);
    if (!pvD->IsReplicated())
      getTouch(pvD, leafDepth + 1, copy, touches);
  }

  if (leafDepth > 0)
    fHistory_.BackLevel();
}

G4VPhysicalVolume *PrintG4Touch::getTopPV() {
  return G4TransportationManager::GetTransportationManager()->GetNavigatorForTracking()->GetWorldVolume();
}

G4LogicalVolume *PrintG4Touch::getTopLV() { return theTopPV_->GetLogicalVolume(); }

#include "SimG4Core/Watcher/interface/SimWatcherFactory.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"

DEFINE_SIMWATCHER(PrintG4Touch);
