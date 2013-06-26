#ifndef SimG4CorePrintGeomInfo_PrintMaterialBudgetInfo_H
#define SimG4CorePrintGeomInfo_PrintMaterialBudgetInfo_H

#include "SimG4Core/Watcher/interface/SimWatcher.h"
#include "SimG4Core/Notification/interface/Observer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "G4NavigationHistory.hh"

#include <iostream>
#include <fstream>
#include <vector>
#include <string>

class BeginOfJob;
class BeginOfRun;
class G4LogicalVolume;
class G4VPhysicalVolume;
class G4VSolid;

typedef std::map< G4VPhysicalVolume*, G4VPhysicalVolume*, std::less<G4VPhysicalVolume*> > mpvpv;
typedef std::multimap< G4LogicalVolume*, G4VPhysicalVolume*, std::less<G4LogicalVolume*> > mmlvpv;

class PrintMaterialBudgetInfo : public SimWatcher,
				public Observer<const BeginOfJob*>,
				public Observer<const BeginOfRun*> {

public:
  PrintMaterialBudgetInfo(edm::ParameterSet const & p);
  ~PrintMaterialBudgetInfo();
private:
  void update(const BeginOfJob* job) {};
  void update(const BeginOfRun* run);
  void dumpHeader(std::ostream& out = std::cout);
  void dumpLaTeXHeader(std::ostream& out = std::cout);
  void dumpHierarchyLeaf(G4VPhysicalVolume* pv, G4LogicalVolume* lv, unsigned int leafDepth,
			 std::ostream& weightOut = std::cout, std::ostream& texOut = std::cout);
  void printInfo(G4VPhysicalVolume* pv, G4LogicalVolume* lv, unsigned int leafDepth,
		 std::ostream& weightOut = std::cout, std::ostream& texOut = std::cout);
  void dumpElementMassFraction(std::ostream& elementOut = std::cout);
  void dumpLaTeXFooter(std::ostream& out = std::cout);
  
private:
  std::string              name;
  int                      nchar;
  mpvpv                    thePVTree;
  G4VPhysicalVolume*       theTopPV; 
  G4NavigationHistory      fHistory;
  bool                     volumeFound;
  unsigned int             levelFound;
  std::ofstream            weightOutputFile;
  std::ofstream            elementOutputFile;
  std::ofstream            texOutputFile;
  std::vector<std::string> elementNames;
  std::vector<double>      elementTotalWeight;
  std::vector<double>      elementWeightFraction;
  //
  std::string stringLaTeXUnderscore(std::string stringname);
  std::string stringLaTeXSuperscript(std::string stringname);
};

#endif
