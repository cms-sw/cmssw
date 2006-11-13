#ifndef Validation_CheckOverlap_H
#define Validation_CheckOverlap_H
#include "SimG4Core/Watcher/interface/SimWatcher.h"
#include "SimG4Core/Notification/interface/Observer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <iostream>
#include <vector>
#include <map>
#include <string>

class BeginOfJob;
class BeginOfRun;
class G4LogicalVolume;
class G4VPhysicalVolume;

typedef std::multimap< G4LogicalVolume*, G4VPhysicalVolume*, std::less<G4LogicalVolume*> > mmlvpv;

class CheckOverlap : public SimWatcher,
		     public Observer<const BeginOfRun *> {

public:

  CheckOverlap(edm::ParameterSet const & p);
  ~CheckOverlap();

private:

  void update(const BeginOfRun * run);
  void checkHierarchyLeafPVLV(G4LogicalVolume * lv, uint leafDepth);
  void checkPV(G4VPhysicalVolume * pv, uint leafDepth);
  G4VPhysicalVolume * getTopPV();

private:

  std::string              nodeName;
  int                      nPoints;
  G4LogicalVolume *        topLV; 

};

#endif
