#include "Validation/Geometry/interface/MaterialBudgetCategorizer.h"
//#include "Utilities/UI/interface/SimpleConfigurable.h"

#include "G4LogicalVolumeStore.hh"
#include "G4Material.hh"

// rr
#include "FWCore/MessageLogger/interface/MessageLogger.h"
// rr

#include<fstream.h>
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
    theMaterialMap[ (*matTable)[ii]->GetName()] = ii++;
  }
  
  // rr
  //----- Build map material name - x0 contributes
  ifstream theMaterialFile;
  theMaterialFile.open("../data/trackerMaterials.x0");
  // fill everything as "other"
  float sup,sen,cab,col,ele,oth,air;
  for( ii = 0; ii < matSize; ii++ ) {
    sup=sen=cab=col=ele=0.;
    oth=1.;
    air=0;
    cout << " material " << (*matTable)[ii]->GetName() << " prepared X0 = " << (*matTable)[ii]->GetRadlen() << " mm" << endl;
    if((*matTable)[ii]->GetName()=="Air") {
      air=1.000; 
      oth=0.000;
    }
    // actually this class does not work if there are spaces into material names --> Recover material properties
    if((*matTable)[ii]->GetName()=="Carbon fibre str.") { 
      sup=1.000; 
      oth=0.000;
    }
    theX0Map[ (*matTable)[ii]->GetName() ].push_back(sup); // sup
    theX0Map[ (*matTable)[ii]->GetName() ].push_back(sen); // sen
    theX0Map[ (*matTable)[ii]->GetName() ].push_back(cab); // cab
    theX0Map[ (*matTable)[ii]->GetName() ].push_back(col); // col
    theX0Map[ (*matTable)[ii]->GetName() ].push_back(ele); // ele
    theX0Map[ (*matTable)[ii]->GetName() ].push_back(oth); // oth
    theX0Map[ (*matTable)[ii]->GetName() ].push_back(air); // air
  }
  //
  
  //
  std::string materialName;
  while(!theMaterialFile.eof()) {
    theMaterialFile >> materialName;
    theMaterialFile >> sup >> sen >> cab >> col >> ele;
    oth = 0.000;
    air = 0.000;
    theX0Map[materialName].clear();        // clear before re-filling
    theX0Map[materialName].push_back(sup); // sup
    theX0Map[materialName].push_back(sen); // sen
    theX0Map[materialName].push_back(cab); // cab
    theX0Map[materialName].push_back(col); // col
    theX0Map[materialName].push_back(ele); // ele
    theX0Map[materialName].push_back(oth); // oth
    theX0Map[materialName].push_back(air); // air
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
  
  // summary of all the materials loaded
  cout << endl << endl << "MaterialBudgetCategorizer::Material Summary --------" << endl;
  for( ii = 0; ii < matSize; ii++ ) {
    //    edm::LogInfo("MaterialBudgetCategorizer")
    cout << " material " << (*matTable)[ii]->GetName()
	 << " SUP " << theX0Map[ (*matTable)[ii]->GetName() ][0] 
	 << " SEN " << theX0Map[ (*matTable)[ii]->GetName() ][1]
	 << " CAB " << theX0Map[ (*matTable)[ii]->GetName() ][2]
	 << " COL " << theX0Map[ (*matTable)[ii]->GetName() ][3]
	 << " ELE " << theX0Map[ (*matTable)[ii]->GetName() ][4]
	 << " OTH " << theX0Map[ (*matTable)[ii]->GetName() ][5]
	 << " AIR " << theX0Map[ (*matTable)[ii]->GetName() ][6]
	 << endl;
  }
  //
  // rr
  
}
