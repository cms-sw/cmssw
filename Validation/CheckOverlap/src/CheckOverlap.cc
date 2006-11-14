#include "Validation/CheckOverlap/interface/CheckOverlap.h"

#include "SimG4Core/Notification/interface/BeginOfJob.h"
#include "SimG4Core/Notification/interface/BeginOfRun.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "G4Run.hh"
#include "G4PhysicalVolumeStore.hh"
#include "G4PVPlacement.hh"
#include "G4PVParameterised.hh"
#include "G4LogicalVolume.hh"
#include "G4TransportationManager.hh"

#include <set>

CheckOverlap::CheckOverlap(const edm::ParameterSet &p) : topLV(0) {
  nodeName = p.getUntrackedParameter<std::string>("NodeName", "");
  nPoints  = p.getUntrackedParameter<int>("Resolution", 1000);
  std::cout << "CheckOverlap:: initialised with Node Name " << " " << nodeName
	    << " and Resolution " << nPoints << std::endl;
}
 
CheckOverlap::~CheckOverlap() {}
  
void CheckOverlap::update(const BeginOfRun * run) {
  
  std::cout << "Node Name " << nodeName << std::endl;
  if (nodeName != "") {
    const G4PhysicalVolumeStore * pvs = G4PhysicalVolumeStore::GetInstance();
    std::vector<G4VPhysicalVolume *>::const_iterator pvcite;
    int i = 0;
    for (pvcite = pvs->begin(); pvcite != pvs->end(); pvcite++) {
      std::cout << "Name of node " << (++i) << " : " << (*pvcite)->GetName()
		<< std::endl;
      if ((*pvcite)->GetName() == (G4String)(nodeName)) {
	topLV = (*pvcite)->GetLogicalVolume();
	break;
      }
    }
  } else {
    G4VPhysicalVolume * theTopPV = getTopPV();
    topLV = theTopPV->GetLogicalVolume();
  }
  std::cout << "Top LV " << topLV;
  if (topLV != 0) std::cout << " Name " << topLV->GetName();
  std::cout << std::endl;
  //---------- Check all PV's
  if (topLV) 
    checkHierarchyLeafPVLV(topLV, 0);
}

void CheckOverlap::checkHierarchyLeafPVLV(G4LogicalVolume * lv, 
					  uint leafDepth) {

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

void CheckOverlap::checkPV(G4VPhysicalVolume * pv, uint leafDepth) {

  //----- PV info
#ifndef G4V7
  std::string mother = "World";
  if (pv->GetMotherLogical()) mother = pv->GetMotherLogical()->GetName();
  if (!pv->IsReplicated()) {
    G4PVPlacement* pvplace = dynamic_cast<G4PVPlacement* >(pv);
    G4bool ok = pvplace->CheckOverlaps(nPoints);
    std::cout << "Placed PV " << pvplace->GetName() << " Number "
	      << pvplace->GetCopyNo() << " in mother " << mother 
	      << " at depth " << leafDepth << " Status " << ok << std::endl;
  } else {
    if (pv->GetParameterisation() != 0) {
      G4PVParameterised* pvparam = dynamic_cast<G4PVParameterised* >(pv);
      G4bool ok = pvparam->CheckOverlaps(nPoints);
      std::cout << "Parametrized PV " << pvparam->GetName() << " in mother "
		<< mother << " at depth " << leafDepth << " Status "	<< ok 
		<< std::endl;
    }
  }
#endif
}

G4VPhysicalVolume * CheckOverlap::getTopPV() {
  return G4TransportationManager::GetTransportationManager()
    ->GetNavigatorForTracking()->GetWorldVolume();
}
