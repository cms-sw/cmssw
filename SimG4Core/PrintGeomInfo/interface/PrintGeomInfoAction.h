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

typedef std::multimap<G4LogicalVolume*, G4VPhysicalVolume*, std::less<G4LogicalVolume*> > mmlvpv;

class PrintGeomInfoAction : public SimWatcher, public Observer<const BeginOfJob*>, public Observer<const BeginOfRun*> {
public:
  PrintGeomInfoAction(edm::ParameterSet const& p);
  ~PrintGeomInfoAction() override;

private:
  void update(const BeginOfJob* job) override;
  void update(const BeginOfRun* run) override;
  void dumpSummary(std::ostream& out = std::cout);
  void dumpG4LVList(std::ostream& out = std::cout);
  void dumpG4LVTree(std::ostream& out = std::cout);
  void dumpMaterialList(std::ostream& out = std::cout);
  void dumpG4LVLeaf(G4LogicalVolume* lv, unsigned int leafDepth, unsigned int count, std::ostream& out = std::cout);
  int countNoTouchables();
  void add1touchable(G4LogicalVolume* lv, int& nTouch);
  void dumpHierarchyTreePVLV(std::ostream& out = std::cout);
  void dumpHierarchyLeafPVLV(G4LogicalVolume* lv, unsigned int leafDepth, std::ostream& out = std::cout);
  void dumpLV(G4LogicalVolume* lv, unsigned int leafDepth, std::ostream& out = std::cout);
  void dumpPV(G4VPhysicalVolume* pv, unsigned int leafDepth, std::ostream& out = std::cout);
  void dumpSolid(G4VSolid* sol, unsigned int leafDepth, std::ostream& out = std::cout);
  void dumpTouch(G4VPhysicalVolume* pv, unsigned int leafDepth, std::ostream& out = std::cout);
  void dumpInFile();
  void getTouch(G4VPhysicalVolume* pv, unsigned int leafDepth, unsigned int copym, std::vector<std::string>& touches);
  std::string spacesFromLeafDepth(unsigned int leafDepth);
  G4VPhysicalVolume* getTopPV();
  G4LogicalVolume* getTopLV();

private:
  bool dumpSummary_, dumpLVTree_, dumpLVList_, dumpMaterial_;
  bool dumpLV_, dumpSolid_, dumpAtts_, dumpPV_;
  bool dumpRotation_, dumpReplica_, dumpTouch_;
  bool dumpSense_, dd4hep_;
  std::string name_;
  int nchar_;
  std::string fileMat_, fileSolid_, fileLV_, filePV_, fileTouch_;
  bool fileDetail_;
  std::vector<std::string> names_;
  G4VPhysicalVolume* theTopPV_;
  G4NavigationHistory fHistory_;
};

#endif
