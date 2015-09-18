#include "Validation/Geometry/interface/MaterialBudgetCategorizer.h"

#include "G4LogicalVolumeStore.hh"
#include "G4Material.hh"
#include "G4UnitsTable.hh"
#include "G4EmCalculator.hh"
#include "G4UnitsTable.hh"

// rr
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
// rr

#include <fstream>
#include <vector>

using namespace std;

MaterialBudgetCategorizer::MaterialBudgetCategorizer()
{
  buildMaps();
}

void MaterialBudgetCategorizer::buildMaps()
{
  //----- Build map volume name - volume index
  G4LogicalVolumeStore* lvs = G4LogicalVolumeStore::GetInstance();
  G4LogicalVolumeStore::iterator ite;
  int ii = 0;
  for (ite = lvs->begin(); ite != lvs->end(); ite++) {
    theVolumeMap[(*ite)->GetName()] = ii++;
  }
  
  //----- Build map material name - volume index
  const G4MaterialTable* matTable = G4Material::GetMaterialTable();
  G4int matSize = matTable->size();
  for( ii = 0; ii < matSize; ii++ ) {
    theMaterialMap[ (*matTable)[ii]->GetName()] = ii+1;
  }
  
  // rr

  //----- Build map material name
  for( int ii = 0; ii < matSize; ii++ ) {
    float sup,sen,cab,col,ele,oth,air;
    sup=sen=cab=col=ele=0.;
    oth=1.;
    air=0;
    cout << " material " << (*matTable)[ii]->GetName() << " prepared"
	 << endl;
    if((*matTable)[ii]->GetName()=="Air"
       ||
       (*matTable)[ii]->GetName()=="Vacuum"
       ) {
      air=1.000; 
      oth=0.000;
    }
    // actually this class does not work if there are spaces into material names --> Recover material properties
    if((*matTable)[ii]->GetName()=="Carbon fibre str.") { 
      sup=1.000; 
      oth=0.000;
    }
    // X0
    theX0Map[ (*matTable)[ii]->GetName() ].push_back(sup); // sup
    theX0Map[ (*matTable)[ii]->GetName() ].push_back(sen); // sen
    theX0Map[ (*matTable)[ii]->GetName() ].push_back(cab); // cab
    theX0Map[ (*matTable)[ii]->GetName() ].push_back(col); // col
    theX0Map[ (*matTable)[ii]->GetName() ].push_back(ele); // ele
    theX0Map[ (*matTable)[ii]->GetName() ].push_back(oth); // oth
    theX0Map[ (*matTable)[ii]->GetName() ].push_back(air); // air
    // L0
    theL0Map[ (*matTable)[ii]->GetName() ].push_back(sup); // sup
    theL0Map[ (*matTable)[ii]->GetName() ].push_back(sen); // sen
    theL0Map[ (*matTable)[ii]->GetName() ].push_back(cab); // cab
    theL0Map[ (*matTable)[ii]->GetName() ].push_back(col); // col
    theL0Map[ (*matTable)[ii]->GetName() ].push_back(ele); // ele
    theL0Map[ (*matTable)[ii]->GetName() ].push_back(oth); // oth
    theL0Map[ (*matTable)[ii]->GetName() ].push_back(air); // air
  }
  //
  
  //----- Build map material name - X0 contributes
  cout << endl << endl << "MaterialBudgetCategorizer::Fill X0 Map" << endl;
  std::string theMaterialX0FileName = edm::FileInPath("Validation/Geometry/data/trackerMaterials.x0").fullPath();
  buildCategoryMap(theMaterialX0FileName, theX0Map);
  //----- Build map material name - L0 contributes
  cout << endl << endl << "MaterialBudgetCategorizer::Fill L0 Map" << endl;
  std::string theMaterialL0FileName = edm::FileInPath("Validation/Geometry/data/trackerMaterials.l0").fullPath();
  buildCategoryMap(theMaterialL0FileName, theL0Map);
  // summary of all the materials loaded
  cout << endl << endl << "MaterialBudgetCategorizer::Material Summary --------" << endl;
  G4EmCalculator calc;
  for( ii = 0; ii < matSize; ii++ ) {
    //    edm::LogInfo("MaterialBudgetCategorizer")
    cout << " material " << (*matTable)[ii]->GetName()
	 << endl
	 << "\t density = " << G4BestUnit((*matTable)[ii]->GetDensity(),"Volumic Mass")
	 << endl
	 << "\t X0 = "      << (*matTable)[ii]->GetRadlen()             << " mm"
	 << endl
	 << "\t Energy threshold for photons for 100 mm range = "
	 << G4BestUnit(calc.ComputeEnergyCutFromRangeCut(100, G4String("gamma"), (*matTable)[ii]->GetName()) , "Energy")
	 << endl
	 << " SUP " << theX0Map[ (*matTable)[ii]->GetName() ][0] 
	 << " SEN " << theX0Map[ (*matTable)[ii]->GetName() ][1]
	 << " CAB " << theX0Map[ (*matTable)[ii]->GetName() ][2]
	 << " COL " << theX0Map[ (*matTable)[ii]->GetName() ][3]
	 << " ELE " << theX0Map[ (*matTable)[ii]->GetName() ][4]
	 << " OTH " << theX0Map[ (*matTable)[ii]->GetName() ][5]
	 << " AIR " << theX0Map[ (*matTable)[ii]->GetName() ][6]
	 << endl
	 << "\t Lambda0 = " << (*matTable)[ii]->GetNuclearInterLength() << " mm"
	 << endl
	 << " SUP " << theL0Map[ (*matTable)[ii]->GetName() ][0] 
	 << " SEN " << theL0Map[ (*matTable)[ii]->GetName() ][1]
	 << " CAB " << theL0Map[ (*matTable)[ii]->GetName() ][2]
	 << " COL " << theL0Map[ (*matTable)[ii]->GetName() ][3]
	 << " ELE " << theL0Map[ (*matTable)[ii]->GetName() ][4]
	 << " OTH " << theL0Map[ (*matTable)[ii]->GetName() ][5]
	 << " AIR " << theL0Map[ (*matTable)[ii]->GetName() ][6]
	 << endl;
    if( theX0Map[ (*matTable)[ii]->GetName() ][5] == 1 || theL0Map[ (*matTable)[ii]->GetName() ][5] == 1 )
      std::cout << "WARNING: material with no category: " << (*matTable)[ii]->GetName() << std::endl;
  }
  //
  // rr
  
}

void MaterialBudgetCategorizer::buildCategoryMap(std::string theMaterialFileName, std::map<std::string,std::vector<float> >& theMap) {
  //  const G4MaterialTable* matTable = G4Material::GetMaterialTable();
  //  G4int matSize = matTable->size();
  
  std::ifstream theMaterialFile(theMaterialFileName);
  if (!theMaterialFile) 
    cms::Exception("LogicError") <<" File not found " << theMaterialFileName;
  
  // fill everything as "other"
  float sup,sen,cab,col,ele,oth,air;
  sup=sen=cab=col=ele=0.;
  oth=1.;
  air=0;
  
  //
  std::string materialName;
  while(theMaterialFile) {
    theMaterialFile >> materialName;
    theMaterialFile >> sup >> sen >> cab >> col >> ele;
    oth = 0.000;
    air = 0.000;
    theMap[materialName].clear();        // clear before re-filling
    theMap[materialName].push_back(sup); // sup
    theMap[materialName].push_back(sen); // sen
    theMap[materialName].push_back(cab); // cab
    theMap[materialName].push_back(col); // col
    theMap[materialName].push_back(ele); // ele
    theMap[materialName].push_back(oth); // oth
    theMap[materialName].push_back(air); // air
    cout << " material " << materialName << " filled " 
	 << " SUP " << sup 
	 << " SEN " << sen 
	 << " CAB " << cab 
	 << " COL " << col 
	 << " ELE " << ele 
	 << " OTH " << oth 
	 << " AIR " << air 
	 << endl;
  }
  
}
