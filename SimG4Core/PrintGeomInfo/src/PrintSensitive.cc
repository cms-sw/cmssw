#include "SimG4Core/PrintGeomInfo/interface/PrintSensitive.h"

#include "SimG4Core/Notification/interface/BeginOfRun.h"
#include "SimG4Core/Notification/interface/SimG4Exception.h"

#include "G4Run.hh"
#include "G4VPhysicalVolume.hh"
#include "G4LogicalVolume.hh"
#include "G4TransportationManager.hh"

#include <set>
#include <map>

PrintSensitive::PrintSensitive(const edm::ParameterSet &p) {
  name  = p.getUntrackedParameter<std::string>("Name","*");
  nchar = name.find("*");
  name.assign(name,0,nchar);
  std::cout << "PrintSensitive:: Print position of all Sensitive Touchables: "
	    << " for names (0-" << nchar << ") = " << name << "\n";
}
 
PrintSensitive::~PrintSensitive() {}
  
void PrintSensitive::update(const BeginOfRun * run) {

  G4VPhysicalVolume * theTopPV = getTopPV();
  dumpTouch(theTopPV, 0, false, std::cout);
}

void PrintSensitive::dumpTouch(G4VPhysicalVolume * pv, unsigned int leafDepth, 
			       bool printIt, std::ostream & out) {

  if (leafDepth == 0) fHistory.SetFirstEntry(pv);
  else fHistory.NewLevel(pv, kNormal, pv->GetCopyNo());

  G4ThreeVector globalpoint = fHistory.GetTopTransform().Inverse().
    TransformPoint(G4ThreeVector(0,0,0));
  G4LogicalVolume * lv = pv->GetLogicalVolume();

  std::string mother = "World";
  if (pv->GetMotherLogical()) mother = pv->GetMotherLogical()->GetName();
  std::string lvname = lv->GetName();
  lvname.assign(lvname,0,nchar);
  if (lvname == name) printIt = true;

  if (lv->GetSensitiveDetector() && printIt) {
    out << leafDepth << " ### VOLUME = " << lv->GetName() 
	<< " Copy No " << pv->GetCopyNo() << " in " << mother
	<< " global position of centre " << globalpoint << " (r=" 
	<<  globalpoint.perp() << ", phi=" <<  globalpoint.phi()/deg
	<< ")\n";
  }

  int NoDaughters = lv->GetNoDaughters();
  while ((NoDaughters--)>0)  {
    G4VPhysicalVolume * pvD = lv->GetDaughter(NoDaughters);
    if (!pvD->IsReplicated()) dumpTouch(pvD, leafDepth+1, printIt, out);
  }

  if (leafDepth > 0) fHistory.BackLevel();
}

G4VPhysicalVolume * PrintSensitive::getTopPV() {
  return G4TransportationManager::GetTransportationManager()
    ->GetNavigatorForTracking()->GetWorldVolume();
}
