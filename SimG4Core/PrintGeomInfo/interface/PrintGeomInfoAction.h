#ifndef SimG4Core_PrintGeomInfoAction_H
#define SimG4Core_PrintGeomInfoAction_H

#include "SimG4Core/Watcher/interface/SimWatcher.h"
#include "SimG4Core/Notification/interface/Observer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
    
#include "G4NavigationHistory.hh"

#include <iostream>
#include <vector>
#include <map>
#include <string>

class BeginOfJob;
class BeginOfRun;
class G4LogicalVolume;
class G4VPhysicalVolume;
class G4VSolid;

typedef std::map< G4VPhysicalVolume*, G4VPhysicalVolume*, std::less<G4VPhysicalVolume*> > mpvpv;
typedef std::multimap< G4LogicalVolume*, G4VPhysicalVolume*, std::less<G4LogicalVolume*> > mmlvpv;

class PrintGeomInfoAction : public SimWatcher,
			    public Observer<const BeginOfJob *>,
			    public Observer<const BeginOfRun *> {

public:
  PrintGeomInfoAction(edm::ParameterSet const & p);
  ~PrintGeomInfoAction();
private:
  void update(const BeginOfJob * job);
  void update(const BeginOfRun * run);
  void dumpSummary(std::ostream& out = std::cout);
  void dumpG4LVList(std::ostream& out = std::cout);
  void dumpG4LVTree(std::ostream& out = std::cout);
  void dumpMaterialList(std::ostream& out = std::cout);
  void dumpG4LVLeaf(G4LogicalVolume * lv, unsigned int leafDepth, unsigned int count, std::ostream & out = std::cout);
  int countNoTouchables();
  void add1touchable(G4LogicalVolume * lv, int & nTouch);
  void dumpHierarchyTreePVLV(std::ostream& out = std::cout);
  void dumpHierarchyLeafPVLV(G4LogicalVolume * lv, unsigned int leafDepth, std::ostream & out = std::cout);
  void dumpLV(G4LogicalVolume * lv, unsigned int leafDepth, std::ostream & out = std::cout);
  void dumpPV(G4VPhysicalVolume * pv, unsigned int leafDepth, std::ostream & out = std::cout);
  void dumpTouch(G4VPhysicalVolume * pv, unsigned int leafDepth, std::ostream & out = std::cout);
  std::string spacesFromLeafDepth(unsigned int leafDepth);
  void dumpSolid(G4VSolid * sol, unsigned int leafDepth, std::ostream & out = std::cout);
  G4VPhysicalVolume * getTopPV();
  G4LogicalVolume * getTopLV();
private:
  bool                     _dumpSummary, _dumpLVTree, _dumpLVList;
  bool                     _dumpMaterial;
  bool                     _dumpLV, _dumpSolid, _dumpAtts, _dumpSense;
  bool                     _dumpPV, _dumpRotation, _dumpReplica, _dumpTouch;
  std::string              name;
  int                      nchar;
  std::vector<std::string> names;
  mpvpv                    thePVTree;
  G4VPhysicalVolume *      theTopPV; 
  G4NavigationHistory      fHistory;
};

#endif
