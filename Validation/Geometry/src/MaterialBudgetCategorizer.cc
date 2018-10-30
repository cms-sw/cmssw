#include "Validation/Geometry/interface/MaterialBudgetCategorizer.h"

#include "G4LogicalVolumeStore.hh"
#include "G4Material.hh"
#include "G4UnitsTable.hh"
#include "G4EmCalculator.hh"
#include "G4UnitsTable.hh"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <fstream>
#include <vector>

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
  
  //----- Build map material name
  for( int ii = 0; ii < matSize; ii++ ) {
    float sup,sen,cab,col,ele,oth,air;
    sup=sen=cab=col=ele=0.;
    oth=1.;
    air=0;
    edm::LogInfo("MaterialBudget") << "MaterialBudgetCategorizer: Material " << (*matTable)[ii]->GetName() << " prepared";
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
  
  //----- Build map material name - X0 contributes
  edm::LogInfo("MaterialBudget") << "MaterialBudgetCategorizer: Fill X0 Map";
  std::string theMaterialX0FileName = edm::FileInPath("Validation/Geometry/data/trackerMaterials.x0").fullPath();
  buildCategoryMap(theMaterialX0FileName, theX0Map);
  //For the HGCal
  std::string thehgcalMaterialX0FileName = edm::FileInPath("Validation/Geometry/data/hgcalMaterials.x0").fullPath();
  buildHGCalCategoryMap(thehgcalMaterialX0FileName, theHGCalX0Map);
  //----- Build map material name - L0 contributes
  edm::LogInfo("MaterialBudget") << "MaterialBudgetCategorizer: Fill L0 Map";
  std::string theMaterialL0FileName = edm::FileInPath("Validation/Geometry/data/trackerMaterials.l0").fullPath();
  buildCategoryMap(theMaterialL0FileName, theL0Map);
  //For the HGCal
  std::string thehgcalMaterialL0FileName = edm::FileInPath("Validation/Geometry/data/hgcalMaterials.l0").fullPath();
  buildHGCalCategoryMap(thehgcalMaterialL0FileName, theHGCalL0Map);
  // summary of all the materials loaded
  edm::LogInfo("MaterialBudget") << "MaterialBudgetCategorizer: Material Summary Starts";
  G4EmCalculator calc;
  for( ii = 0; ii < matSize; ii++ ) {
    edm::LogInfo("MaterialBudget") 
      << "MaterialBudgetCateogorizer: Material " << (*matTable)[ii]->GetName()
      << std::endl
      << "\t density = " << G4BestUnit((*matTable)[ii]->GetDensity(),"Volumic Mass")
      << std::endl
      << "\t X0 = "      << (*matTable)[ii]->GetRadlen()             << " mm"
      << std::endl
      << "\t Energy threshold for photons for 100 mm range = "
      << G4BestUnit(calc.ComputeEnergyCutFromRangeCut(100, G4String("gamma"), (*matTable)[ii]->GetName()) , "Energy")
      << std::endl
      << " SUP " << theX0Map[ (*matTable)[ii]->GetName() ][0] 
      << " SEN " << theX0Map[ (*matTable)[ii]->GetName() ][1]
      << " CAB " << theX0Map[ (*matTable)[ii]->GetName() ][2]
      << " COL " << theX0Map[ (*matTable)[ii]->GetName() ][3]
      << " ELE " << theX0Map[ (*matTable)[ii]->GetName() ][4]
      << " OTH " << theX0Map[ (*matTable)[ii]->GetName() ][5]
      << " AIR " << theX0Map[ (*matTable)[ii]->GetName() ][6]
      << std::endl
      << "\t Lambda0 = " << (*matTable)[ii]->GetNuclearInterLength() << " mm"
      << std::endl
      << " SUP " << theL0Map[ (*matTable)[ii]->GetName() ][0] 
      << " SEN " << theL0Map[ (*matTable)[ii]->GetName() ][1]
      << " CAB " << theL0Map[ (*matTable)[ii]->GetName() ][2]
      << " COL " << theL0Map[ (*matTable)[ii]->GetName() ][3]
      << " ELE " << theL0Map[ (*matTable)[ii]->GetName() ][4]
      << " OTH " << theL0Map[ (*matTable)[ii]->GetName() ][5]
      << " AIR " << theL0Map[ (*matTable)[ii]->GetName() ][6]
      << std::endl;
    if( theX0Map[ (*matTable)[ii]->GetName() ][5] == 1 || theL0Map[ (*matTable)[ii]->GetName() ][5] == 1 )
      edm::LogWarning("MaterialBudget") 
	<< "MaterialBudgetCategorizer Material with no category: " << (*matTable)[ii]->GetName();
  }
}

void MaterialBudgetCategorizer::buildCategoryMap(std::string theMaterialFileName, std::map<std::string,std::vector<float> >& theMap) {
  
  std::ifstream theMaterialFile(theMaterialFileName);
  if (!theMaterialFile) 
    cms::Exception("LogicError") << " File not found " << theMaterialFileName;
  
  // fill everything as "other"
  float sup,sen,cab,col,ele,oth,air;
  sup=sen=cab=col=ele=0.;

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
    LogDebug("MaterialBudget") 
      << "MaterialBudgetCategorizer: Material " << materialName << " filled " 
      << " SUP " << sup 
      << " SEN " << sen 
      << " CAB " << cab 
      << " COL " << col 
      << " ELE " << ele 
      << " OTH " << oth 
      << " AIR " << air;
  }
}

void MaterialBudgetCategorizer::buildHGCalCategoryMap(std::string theMaterialFileName, 
						      std::map<std::string,std::vector<float> >& theMap)
{
  
  std::ifstream theMaterialFile(theMaterialFileName);
  if (!theMaterialFile) 
    cms::Exception("LogicError") <<" File not found " << theMaterialFileName;
  
  // fill everything as "other"
  float Air,Cables,Copper,H_Scintillator,Lead,M_NEMA_FR4_plate,Silicon,StainlessSteel,WCu, oth; 
  Air=Cables=Copper=H_Scintillator=Lead=M_NEMA_FR4_plate=Silicon=StainlessSteel=WCu=0.;

  std::string materialName;
  while(theMaterialFile) {
    theMaterialFile >> materialName;
    theMaterialFile >> Air >> Cables >> Copper >> H_Scintillator >> Lead >> M_NEMA_FR4_plate >> Silicon >> StainlessSteel >> WCu;
    // Skip comments
    if (materialName[0] == '#')
      continue;
    // Substitute '*' with spaces
    std::replace(materialName.begin(), materialName.end(), '*', ' ');
    oth = 0.000;
    theMap[materialName].clear();        // clear before re-filling
    theMap[materialName].push_back(Air             ); // Air
    theMap[materialName].push_back(Cables          ); // Cables          
    theMap[materialName].push_back(Copper          ); // Copper          
    theMap[materialName].push_back(H_Scintillator  ); // H_Scintillator  
    theMap[materialName].push_back(Lead            ); // Lead            
    theMap[materialName].push_back(M_NEMA_FR4_plate); // M_NEMA_FR4_plate
    theMap[materialName].push_back(Silicon         ); // Silicon         
    theMap[materialName].push_back(StainlessSteel  ); // StainlessSteel
    theMap[materialName].push_back(WCu             ); // WCu
    theMap[materialName].push_back(oth             ); // oth
    edm::LogInfo("MaterialBudget") 
      << "MaterialBudgetCategorizer: material " << materialName << " filled " 
      << std::endl
      << "\tAir              " << Air << std::endl
      << "\tCables           " << Cables << std::endl
      << "\tCopper           " << Copper << std::endl
      << "\tH_Scintillator   " << H_Scintillator << std::endl
      << "\tLead             " << Lead << std::endl
      << "\tM_NEMA_FR4_plate " << M_NEMA_FR4_plate << std::endl
      << "\tSilicon          " << Silicon << std::endl
      << "\tStainlessSteel   " << StainlessSteel << std::endl
      << "\tWCu              " << WCu << std::endl
      << "\tOTH              " << oth;
  }

}
