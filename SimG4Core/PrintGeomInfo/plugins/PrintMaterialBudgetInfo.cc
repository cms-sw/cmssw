#include "SimG4Core/Notification/interface/BeginOfJob.h"
#include "SimG4Core/Notification/interface/BeginOfRun.h"
#include "SimG4Core/Notification/interface/Observer.h"
#include "SimG4Core/Watcher/interface/SimWatcher.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDFilter.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "DetectorDescription/Core/interface/DDValue.h"

#include "G4Run.hh"
#include "G4PhysicalVolumeStore.hh"
#include "G4LogicalVolumeStore.hh"
#include "G4VPhysicalVolume.hh"
#include "G4LogicalVolume.hh"
#include "G4VSolid.hh"
#include "G4Material.hh"
#include "G4NavigationHistory.hh"
#include "G4Track.hh"
#include "G4VisAttributes.hh"
#include "G4UserLimits.hh"
#include "G4TransportationManager.hh"
#include "G4UnitsTable.hh"
#include "Randomize.hh"

#include <iostream>
#include <fstream>
#include <memory>
#include <set>
#include <vector>
#include <string>

typedef std::map<G4VPhysicalVolume*, G4VPhysicalVolume*, std::less<G4VPhysicalVolume*> > mpvpv;
typedef std::multimap<G4LogicalVolume*, G4VPhysicalVolume*, std::less<G4LogicalVolume*> > mmlvpv;

class PrintMaterialBudgetInfo : public SimWatcher,
                                public Observer<const BeginOfJob*>,
                                public Observer<const BeginOfRun*> {
public:
  PrintMaterialBudgetInfo(edm::ParameterSet const& p);
  ~PrintMaterialBudgetInfo() override;

private:
  void update(const BeginOfJob* job) override {}
  void update(const BeginOfRun* run) override;
  void dumpHeader(std::ostream& out = G4cout);
  void dumpLaTeXHeader(std::ostream& out = G4cout);
  void dumpHierarchyLeaf(G4VPhysicalVolume* pv,
                         G4LogicalVolume* lv,
                         unsigned int leafDepth,
                         std::ostream& weightOut = G4cout,
                         std::ostream& texOut = G4cout);
  void printInfo(G4VPhysicalVolume* pv,
                 G4LogicalVolume* lv,
                 unsigned int leafDepth,
                 std::ostream& weightOut = G4cout,
                 std::ostream& texOut = G4cout);
  void dumpElementMassFraction(std::ostream& elementOut = G4cout);
  void dumpLaTeXFooter(std::ostream& out = G4cout);

private:
  std::string name;
  int nchar;
  mpvpv thePVTree;
  G4NavigationHistory fHistory;
  bool volumeFound;
  unsigned int levelFound;
  std::ofstream weightOutputFile;
  std::ofstream elementOutputFile;
  std::ofstream texOutputFile;
  std::vector<std::string> elementNames;
  std::vector<double> elementTotalWeight;
  std::vector<double> elementWeightFraction;

  std::string stringLaTeXUnderscore(std::string stringname);
  std::string stringLaTeXSuperscript(std::string stringname);
};

PrintMaterialBudgetInfo::PrintMaterialBudgetInfo(const edm::ParameterSet& p) {
  name = p.getUntrackedParameter<std::string>("Name", "*");
  nchar = name.find('*');
  name.assign(name, 0, nchar);
  G4cout << "PrintMaterialBudget selected volume " << name << G4endl;
  volumeFound = false;
  std::string weightFileName = name + ".weight";
  weightOutputFile.open(weightFileName.c_str());
  std::string elementFileName = name + ".element";
  elementOutputFile.open(elementFileName.c_str());
  std::string texFileName = name + "_table.tex";
  texOutputFile.open(texFileName.c_str());
  G4cout << "PrintMaterialBudget output file " << weightFileName << G4endl;
  G4cout << "PrintMaterialBudget output file " << elementFileName << G4endl;
  G4cout << "PrintMaterialBudget output file " << texFileName << G4endl;
  elementNames.clear();
  elementTotalWeight.clear();
  elementWeightFraction.clear();
}

PrintMaterialBudgetInfo::~PrintMaterialBudgetInfo() {}

void PrintMaterialBudgetInfo::update(const BeginOfRun* run) {
  [[clang::suppress]] G4Random::setTheEngine(new CLHEP::RanecuEngine);
  // Physical Volume
  G4VPhysicalVolume* theTopPV =
      G4TransportationManager::GetTransportationManager()->GetNavigatorForTracking()->GetWorldVolume();
  assert(theTopPV);
  // Logical Volume
  G4LogicalVolume* lv = theTopPV->GetLogicalVolume();
  unsigned int leafDepth = 0;
  // the first time fill the vectors of elements
  if (elementNames.empty() && elementTotalWeight.empty() && elementWeightFraction.empty()) {
    for (unsigned int iElement = 0; iElement < G4Element::GetNumberOfElements();
         iElement++) {  // first element in table is 0
      elementNames.push_back("rr");
      elementTotalWeight.push_back(0);
      elementWeightFraction.push_back(0);
    }
  }
  dumpHeader(weightOutputFile);
  dumpLaTeXHeader(texOutputFile);
  dumpHierarchyLeaf(theTopPV, lv, leafDepth, weightOutputFile, texOutputFile);
  dumpElementMassFraction(elementOutputFile);
  dumpLaTeXFooter(texOutputFile);
  //
}

void PrintMaterialBudgetInfo::dumpHeader(std::ostream& out) {
  out << "Geom."
      << "\t"
      << "Volume"
      << "\t"
      << "\t"
      << "Copy"
      << "\t"
      << "Solid"
      << "\t"
      << "\t"
      << "Material"
      << "\t"
      << "Density"
      << "\t"
      << "\t"
      << "Mass"
      << "\t"
      << "\t" << G4endl;
  out << "Level"
      << "\t"
      << "Name"
      << "\t"
      << "\t"
      << "Number"
      << "\t"
      << "Name"
      << "\t"
      << "\t"
      << "Name"
      << "\t"
      << "\t"
      << "[g/cm3]"
      << "\t"
      << "\t"
      << "[g]    "
      << "\t"
      << "\t" << G4endl;
}

void PrintMaterialBudgetInfo::dumpLaTeXHeader(std::ostream& out) {
  out << "\\begin{table}[h!]" << G4endl << "  \\caption{\\textsf {" << name << "} volume list.}" << G4endl
      << "  \\label{tab: " << name << "}" << G4endl << "  \\begin{center}" << G4endl << "    \\begin{tabular}{ccccccc}"
      << G4endl << "      \\hline" << G4endl;
  out << "      Geom."
      << "\t & "
      << "      Volume"
      << "\t & "
      << "      Copy"
      << "\t & "
      << "      Solid"
      << "\t & "
      << "      Material"
      << "\t & "
      << "      Density"
      << "\t & "
      << "      Mass"
      << "\t \\\\ " << G4endl;
  out << "      Level"
      << "\t & "
      << "      Name"
      << "\t & "
      << "      Number"
      << "\t & "
      << "      Name"
      << "\t & "
      << "      Name"
      << "\t & "
      << "                "
      << "\t & "
      << "                "
      << "\t \\\\ " << G4endl << "      \\hline\\hline" << G4endl;
}

void PrintMaterialBudgetInfo::dumpLaTeXFooter(std::ostream& out) {
  out << "      \\hline" << G4endl << "    \\end{tabular}" << G4endl << "  \\end{center}" << G4endl << "\\end{table}"
      << G4endl;
}

void PrintMaterialBudgetInfo::dumpHierarchyLeaf(
    G4VPhysicalVolume* pv, G4LogicalVolume* lv, unsigned int leafDepth, std::ostream& weightOut, std::ostream& texOut) {
  if (volumeFound && (leafDepth <= levelFound))
    return;
  if (volumeFound && (leafDepth > levelFound))
    printInfo(pv, lv, leafDepth, weightOut, texOut);

  // choose mother volume
  std::string lvname = lv->GetName();
  lvname.assign(lvname, 0, nchar);
  if (lvname == name) {
    volumeFound = true;
    levelFound = leafDepth;
    printInfo(pv, lv, leafDepth, weightOut, texOut);
    texOut << "      \\hline" << G4endl;
  }

  //----- Get LV daughters from list of PV daughters
  mmlvpv lvpvDaughters;
  std::set<G4LogicalVolume*> lvDaughters;
  int NoDaughters = lv->GetNoDaughters();
  while ((NoDaughters--) > 0) {
    G4VPhysicalVolume* pvD = lv->GetDaughter(NoDaughters);
    lvpvDaughters.insert(mmlvpv::value_type(pvD->GetLogicalVolume(), pvD));
    lvDaughters.insert(pvD->GetLogicalVolume());
  }

  std::set<G4LogicalVolume*>::const_iterator scite;
  mmlvpv::const_iterator mmcite;

  //----- Dump daughters PV and LV
  for (scite = lvDaughters.begin(); scite != lvDaughters.end(); scite++) {
    std::pair<mmlvpv::iterator, mmlvpv::iterator> mmER = lvpvDaughters.equal_range(*scite);
    //----- Dump daughters PV of this LV
    for (mmcite = mmER.first; mmcite != mmER.second; mmcite++)
      dumpHierarchyLeaf((*mmcite).second, *scite, leafDepth + 1, weightOut, texOut);
  }
}

void PrintMaterialBudgetInfo::printInfo(
    G4VPhysicalVolume* pv, G4LogicalVolume* lv, unsigned int leafDepth, std::ostream& weightOut, std::ostream& texOut) {
  double density = lv->GetMaterial()->GetDensity();
  double weight = lv->GetMass(false, false);

  std::string volumeName = lv->GetName();
  if (volumeName.size() < 8)
    volumeName.append("\t");

  std::string solidName = lv->GetSolid()->GetName();
  if (solidName.size() < 8)
    solidName.append("\t");

  std::string materialName = lv->GetMaterial()->GetName();
  if (materialName.size() < 8)
    materialName.append("\t");

  //----- dump info
  weightOut << leafDepth << "\t" << volumeName << "\t" << pv->GetCopyNo() << "\t" << solidName << "\t" << materialName
            << "\t" << G4BestUnit(density, "Volumic Mass") << "\t" << G4BestUnit(weight, "Mass") << "\t" << G4endl;
  //
  texOut << "\t" << leafDepth << "\t & " << stringLaTeXUnderscore(volumeName) << "\t & " << pv->GetCopyNo() << "\t & "
         << stringLaTeXUnderscore(solidName) << "\t & " << stringLaTeXUnderscore(materialName) << "\t & "
         << stringLaTeXSuperscript(G4BestUnit(density, "Volumic Mass")) << "\t & "
         << stringLaTeXSuperscript(G4BestUnit(weight, "Mass")) << "\t \\\\ " << G4endl;
  //
  for (unsigned int iElement = 0; iElement < (unsigned int)lv->GetMaterial()->GetNumberOfElements(); iElement++) {
    // exclude Air in element weight fraction computation
    if (materialName.find("Air")) {
      std::string elementName = lv->GetMaterial()->GetElement(iElement)->GetName();
      double elementMassFraction = lv->GetMaterial()->GetFractionVector()[iElement];
      double elementWeight = weight * elementMassFraction;
      unsigned int elementIndex = (unsigned int)lv->GetMaterial()->GetElement(iElement)->GetIndex();
      elementNames[elementIndex] = elementName;
      elementTotalWeight[elementIndex] += elementWeight;
    }
  }
}

void PrintMaterialBudgetInfo::dumpElementMassFraction(std::ostream& elementOut) {
  // calculate mass fraction
  double totalWeight = 0.0;
  double totalFraction = 0.0;
  for (unsigned int iElement = 0; iElement < (unsigned int)elementTotalWeight.size(); iElement++) {
    totalWeight += elementTotalWeight[iElement];
  }
  // calculate element mass fractions
  for (unsigned int iElement = 0; iElement < (unsigned int)elementTotalWeight.size(); iElement++) {
    elementWeightFraction[iElement] = elementTotalWeight[iElement] / totalWeight;
    totalFraction += elementWeightFraction[iElement];
  }
  // header
  elementOut << "Element"
             << "\t\t"
             << "Index"
             << "\t"
             << "Total Mass"
             << "\t"
             << "Mass Fraction "
             << "\t" << G4endl;
  // dump
  for (unsigned int iElement = 0; iElement < (unsigned int)elementTotalWeight.size(); iElement++) {
    if (elementNames[iElement] != "rr") {
      if (elementNames[iElement].size() < 8)
        elementNames[iElement].append("\t");
      elementOut << elementNames[iElement] << "\t" << iElement << "\t"
                 << G4BestUnit(elementTotalWeight[iElement], "Mass") << "\t" << elementWeightFraction[iElement]
                 << G4endl;
    }
  }
  elementOut << "\n\t\tTotal Weight without Air " << G4BestUnit(totalWeight, "Mass") << "\tTotal Fraction "
             << totalFraction << G4endl;
}

std::string PrintMaterialBudgetInfo::stringLaTeXUnderscore(std::string stringname) {
  // To replace '\' with '\_' to compile LaTeX output
  std::string stringoutput;

  for (unsigned int i = 0; i < stringname.length(); i++) {
    if (stringname.substr(i, 1) == "_") {
      stringoutput += "\\_";
    } else {
      stringoutput += stringname.substr(i, 1);
    }
  }

  return stringoutput;
}

std::string PrintMaterialBudgetInfo::stringLaTeXSuperscript(std::string stringname) {
  // To replace 'm3' with 'm$^3$' to compile LaTeX output
  std::string stringoutput = stringname.substr(0, 1);

  for (unsigned int i = 1; i < stringname.length(); i++) {
    if (stringname.substr(i - 1, 1) == "m" && stringname.substr(i, 1) == "3") {
      stringoutput += "$^3$";
    } else {
      stringoutput += stringname.substr(i, 1);
    }
  }

  return stringoutput;
}

#include "SimG4Core/Watcher/interface/SimWatcherFactory.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"

DEFINE_SIMWATCHER(PrintMaterialBudgetInfo);
