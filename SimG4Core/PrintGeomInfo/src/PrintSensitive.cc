#include "SimG4Core/PrintGeomInfo/interface/PrintSensitive.h"

#include "SimG4Core/Notification/interface/BeginOfRun.h"

#include "G4Run.hh"
#include "G4VPhysicalVolume.hh"
#include "G4LogicalVolume.hh"
#include "G4TransportationManager.hh"

#include <set>
#include <map>

PrintSensitive::PrintSensitive(const edm::ParameterSet &p) {
  name_ = p.getUntrackedParameter<std::string>("Name", "*");
  nchar_ = name_.find("*");
  name_.assign(name_, 0, nchar_);
  std::cout << "PrintSensitive:: Print position of all Sensitive Touchables: "
            << " for names (0-" << nchar_ << ") = " << name_ << "\n";
}

PrintSensitive::~PrintSensitive() {}

void PrintSensitive::update(const BeginOfRun *run) {
  G4VPhysicalVolume *theTopPV = getTopPV();
  int nsens = dumpTouch(theTopPV, 0, false, 0, std::cout);
  std::cout << "\nTotal number of sensitive detector volumes for " << name_ << " is " << nsens << std::endl;
}

int PrintSensitive::dumpTouch(G4VPhysicalVolume *pv, unsigned int leafDepth, bool printIt, int ns, std::ostream &out) {
  if (leafDepth == 0)
    fHistory.SetFirstEntry(pv);
  else
    fHistory.NewLevel(pv, kNormal, pv->GetCopyNo());

  int nsens(ns);
  G4ThreeVector globalpoint = fHistory.GetTopTransform().Inverse().TransformPoint(G4ThreeVector(0, 0, 0));
  G4LogicalVolume *lv = pv->GetLogicalVolume();

  std::string mother = "World";
  if (pv->GetMotherLogical())
    mother = pv->GetMotherLogical()->GetName();
  std::string lvname = lv->GetName();
  lvname.assign(lvname, 0, nchar_);
  if (lvname == name_)
    printIt = true;

  if (lv->GetSensitiveDetector() && printIt) {
    ++nsens;
    out << nsens << ":" << leafDepth << " ### VOLUME = " << lv->GetName() << " Copy No " << pv->GetCopyNo() << " in "
        << mother << " global position of centre " << globalpoint << " (r=" << globalpoint.perp()
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
