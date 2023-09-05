#include "SimG4Core/Notification/interface/BeginOfRun.h"
#include "SimG4Core/Notification/interface/Observer.h"
#include "SimG4Core/Watcher/interface/SimWatcher.h"
#include "SimG4Core/Geometry/interface/DD4hep2DDDName.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "G4Run.hh"
#include "G4VPhysicalVolume.hh"
#include "G4LogicalVolume.hh"
#include "G4NavigationHistory.hh"
#include "G4TransportationManager.hh"

#include <iostream>
#include <string>

class PrintSensitive : public SimWatcher, public Observer<const BeginOfRun *> {
public:
  PrintSensitive(edm::ParameterSet const &p);
  ~PrintSensitive() override;

private:
  void update(const BeginOfRun *run) override;
  int dumpTouch(G4VPhysicalVolume *pv, unsigned int leafDepth, bool printIt, int ns, std::ostream &out = G4cout);
  G4VPhysicalVolume *getTopPV();

private:
  std::string name_;
  bool dd4hep_;
  int nchar_;
  G4NavigationHistory fHistory;
};

PrintSensitive::PrintSensitive(const edm::ParameterSet &p) {
  name_ = p.getParameter<std::string>("Name");
  dd4hep_ = p.getParameter<bool>("DD4hep");
  nchar_ = name_.find('*');
  name_.assign(name_, 0, nchar_);
  G4cout << "PrintSensitive:: Print position of all Sensitive Touchables: "
         << " for names (0-" << nchar_ << ") = " << name_ << " dd4hep " << dd4hep_ << G4endl;
}

PrintSensitive::~PrintSensitive() {}

void PrintSensitive::update(const BeginOfRun *run) {
  G4VPhysicalVolume *theTopPV = getTopPV();
  int nsens = dumpTouch(theTopPV, 0, false, 0, G4cout);
  G4cout << "\nTotal number of sensitive detector volumes for " << name_ << " is " << nsens << G4endl;
}

int PrintSensitive::dumpTouch(G4VPhysicalVolume *pv, unsigned int leafDepth, bool printIt, int ns, std::ostream &out) {
  if (leafDepth == 0)
    fHistory.SetFirstEntry(pv);
  else
    fHistory.NewLevel(pv, kNormal, pv->GetCopyNo());

  int nsens(ns);
  G4ThreeVector globalpoint = fHistory.GetTopTransform().Inverse().TransformPoint(G4ThreeVector(0, 0, 0));
  G4LogicalVolume *lv = pv->GetLogicalVolume();

  std::string mother = (pv->GetMotherLogical())
                           ? (DD4hep2DDDName::nameSolid(
                                 static_cast<std::string>(pv->GetMotherLogical()->GetSolid()->GetName()), dd4hep_))
                           : "World";
  std::string lvname = DD4hep2DDDName::nameSolid(static_cast<std::string>(lv->GetSolid()->GetName()), dd4hep_);
  if (nchar_ > 0) {
    lvname.assign(lvname, 0, nchar_);
    if (lvname == name_)
      printIt = true;
  } else {
    printIt = true;
  }

  if (lv->GetSensitiveDetector() && printIt) {
    ++nsens;
    lvname = DD4hep2DDDName::nameSolid(static_cast<std::string>(lv->GetName()), dd4hep_);
    out << nsens << ":" << leafDepth << " ### VOLUME = " << lvname << " Copy No " << pv->GetCopyNo() << " in " << mother
        << " global position of centre " << globalpoint << " (r=" << globalpoint.perp()
        << ", phi=" << globalpoint.phi() / CLHEP::deg << ")\n";
  }

  int NoDaughters = lv->GetNoDaughters();
  while ((NoDaughters--) > 0) {
    G4VPhysicalVolume *pvD = lv->GetDaughter(NoDaughters);
    if (!pvD->IsReplicated())
      nsens = dumpTouch(pvD, leafDepth + 1, printIt, nsens, out);
  }

  if (leafDepth > 0)
    fHistory.BackLevel();
  return nsens;
}

G4VPhysicalVolume *PrintSensitive::getTopPV() {
  return G4TransportationManager::GetTransportationManager()->GetNavigatorForTracking()->GetWorldVolume();
}

#include "SimG4Core/Watcher/interface/SimWatcherFactory.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"

DEFINE_SIMWATCHER(PrintSensitive);
