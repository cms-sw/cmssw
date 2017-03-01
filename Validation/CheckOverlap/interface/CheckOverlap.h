#ifndef Validation_CheckOverlap_H
#define Validation_CheckOverlap_H
#include "SimG4Core/Watcher/interface/SimWatcher.h"
#include "SimG4Core/Notification/interface/Observer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <iostream>
#include <vector>
#include <map>
#include <string>

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
  void checkHierarchyLeafPVLV(G4LogicalVolume * lv, unsigned int leafDepth);
  void checkPV(G4VPhysicalVolume * pv, unsigned int leafDepth);
  G4VPhysicalVolume * getTopPV();
  void dumpLV(G4LogicalVolume * lv, std::string str);

private:

  std::vector<std::string>      nodeNames;
  int                           nPoints;
  std::vector<G4LogicalVolume*> topLV; 

};

#endif
