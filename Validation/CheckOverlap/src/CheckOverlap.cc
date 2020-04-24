#include "Validation/CheckOverlap/interface/CheckOverlap.h"

#include "SimG4Core/Notification/interface/BeginOfRun.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "G4Run.hh"
#include "G4LogicalVolumeStore.hh"
#include "G4PVPlacement.hh"
#include "G4PVParameterised.hh"
#include "G4LogicalVolume.hh"
#include "G4Material.hh"
#include "G4TransportationManager.hh"

#include <set>

CheckOverlap::CheckOverlap(const edm::ParameterSet &p) : topLV(0) {
  std::vector<std::string> defNames;
  nodeNames = p.getUntrackedParameter<std::vector<std::string> >("NodeNames", defNames);
  nPoints   = p.getUntrackedParameter<int>("Resolution", 1000);
  G4cout << "CheckOverlap:: initialised with " 
	 << nodeNames.size() << " Node Names and Resolution " 
	 << nPoints << " the names are:" << G4endl; 
  for (unsigned int ii=0; ii<nodeNames.size(); ii++)
    G4cout << "CheckOverlap:: Node[" << ii << "] : " << nodeNames[ii] << G4endl; 
}
 
CheckOverlap::~CheckOverlap() {}
  
void CheckOverlap::update(const BeginOfRun * run) {
  
  if (nodeNames.size() > 0) {
    const G4LogicalVolumeStore * lvs = G4LogicalVolumeStore::GetInstance();
    G4cout << "CheckOverlap::update nLV= " << lvs->size() << G4endl; 
    std::vector<G4LogicalVolume *>::const_iterator lvcite;
    int i = 0;
    for (lvcite = lvs->begin(); lvcite != lvs->end(); lvcite++) {
      for (unsigned int ii=0; ii<nodeNames.size(); ii++) {
	if ((*lvcite)->GetName() == (G4String)(nodeNames[ii])) {
	  topLV.push_back((*lvcite));
	  break;
	}
      }
      G4cout << "Name of node " << (++i) << " : " 
	     << (*lvcite)->GetName() << G4endl;
      if (topLV.size() == nodeNames.size()) break;
    }
  } else {
    G4VPhysicalVolume * theTopPV = getTopPV();
    topLV.push_back(theTopPV->GetLogicalVolume());
  }

  if (topLV.size() == 0) {
    G4cout << "No Top LV Found" << G4endl;
  } else {
    for (unsigned int ii=0; ii<topLV.size(); ++ii) {
      G4cout << "Top LV Name " << topLV[ii]->GetName() << G4endl;
      checkHierarchyLeafPVLV(topLV[ii], 0);
    }
  }
}

void CheckOverlap::checkHierarchyLeafPVLV(G4LogicalVolume * lv, 
					  unsigned int leafDepth) {

  //----- Get LV daughters from list of PV daughters
  mmlvpv lvpvDaughters;
  std::set< G4LogicalVolume * > lvDaughters;
  int NoDaughters = lv->GetNoDaughters();
  while ((NoDaughters--)>0) {
    G4VPhysicalVolume * pvD = lv->GetDaughter(NoDaughters);
    lvpvDaughters.insert(mmlvpv::value_type(pvD->GetLogicalVolume(), pvD));
    lvDaughters.insert(pvD->GetLogicalVolume());
  }
 
  std::set< G4LogicalVolume * >::const_iterator scite;
  mmlvpv::const_iterator mmcite;

  //----- Check daughters of LV
  for (scite = lvDaughters.begin(); scite != lvDaughters.end(); scite++) {
    std::pair< mmlvpv::iterator, mmlvpv::iterator > mmER = lvpvDaughters.equal_range(*scite);    
    //----- Check daughters PV of this LV
    for (mmcite = mmER.first ; mmcite != mmER.second; mmcite++) 
      checkPV((*mmcite).second, leafDepth+1);
    //----- Check daughters LV
    checkHierarchyLeafPVLV(*scite, leafDepth+1);
  }
}

void CheckOverlap::checkPV(G4VPhysicalVolume * pv, unsigned int leafDepth) {

  //----- PV info
#ifndef G4V7
  std::string mother = "DDDWorld";
  if (pv->GetMotherLogical()) mother = pv->GetMotherLogical()->GetName();
  if (!pv->IsReplicated()) {
    G4PVPlacement* pvplace = dynamic_cast<G4PVPlacement* >(pv);
    G4bool ok = pvplace->CheckOverlaps(nPoints);
    G4cout << "Placed PV " << pvplace->GetName() 
	   << " Number " << pvplace->GetCopyNo() 
	   << " in mother " << mother << " at depth " 
	   << leafDepth << " Status " << ok << G4endl;
    if (ok) {
      if(pv->GetRotation() == 0) {
	G4cout << "Translation " << pv->GetTranslation()
	       << " and with no rotation" << G4endl;
      } else {
	G4cout << "Translation " << pv->GetTranslation()
	       << " and with rotation "<< *(pv->GetRotation()) << G4endl;
      }
      G4LogicalVolume* lv = pv->GetLogicalVolume();
      dumpLV(lv, "Self");
      if (pv->GetMotherLogical()) {
	lv = pv->GetMotherLogical();
	dumpLV (lv, "Mother");
      }
    }
  } else {
    if (pv->GetParameterisation() != 0) {
      G4PVParameterised* pvparam = dynamic_cast<G4PVParameterised* >(pv);
      G4bool ok = pvparam->CheckOverlaps(nPoints);
      G4cout << "Parametrized PV " << pvparam->GetName()
			     << " in mother " << mother << " at depth "
	     << leafDepth << " Status "	<< ok << G4endl;
    }
  }
#endif
}

G4VPhysicalVolume * CheckOverlap::getTopPV() {
  return G4TransportationManager::GetTransportationManager()
    ->GetNavigatorForTracking()->GetWorldVolume();
}

void CheckOverlap::dumpLV(G4LogicalVolume* lv, std::string str) {
  G4cout << "Dump of " << str << " Logical Volume " 
			 << lv->GetName() << "  Solid: " 
			 << lv->GetSolid()->GetName() << "  Material: "
	 << lv->GetMaterial()->GetName() << G4endl;
  G4cout << *(lv->GetSolid()) << G4endl;
}

