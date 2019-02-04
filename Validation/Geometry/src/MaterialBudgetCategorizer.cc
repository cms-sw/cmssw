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

MaterialBudgetCategorizer::MaterialBudgetCategorizer(std::string mode)
{
  //----- Build map volume name - volume index
  G4LogicalVolumeStore* lvs = G4LogicalVolumeStore::GetInstance();
  G4LogicalVolumeStore::iterator ite;
  int ii = 0;
  for (ite = lvs->begin(); ite != lvs->end(); ite++) {
    theVolumeMap[(*ite)->GetName()] = ii++;
  }

  if ( mode.compare("Tracker") == 0 ) {
    std::string theMaterialX0FileName = edm::FileInPath("Validation/Geometry/data/trackerMaterials.x0").fullPath();
    buildCategoryMap(theMaterialX0FileName, theX0Map);
    std::string theMaterialL0FileName = edm::FileInPath("Validation/Geometry/data/trackerMaterials.l0").fullPath();
    buildCategoryMap(theMaterialL0FileName, theL0Map);
  } else if ( mode.compare("HGCal") == 0 ){
      std::string thehgcalMaterialX0FileName = edm::FileInPath("Validation/Geometry/data/hgcalMaterials.x0").fullPath();
      buildHGCalCategoryMap(thehgcalMaterialX0FileName, theHGCalX0Map);
      std::string thehgcalMaterialL0FileName = edm::FileInPath("Validation/Geometry/data/hgcalMaterials.l0").fullPath();
      buildHGCalCategoryMap(thehgcalMaterialL0FileName, theHGCalL0Map);
  }
}

void MaterialBudgetCategorizer::buildCategoryMap(std::string theMaterialFileName, 
						 std::map<std::string,std::vector<float> >& theMap) 
{

  std::ifstream theMaterialFile(theMaterialFileName);

  if (!theMaterialFile) 
    cms::Exception("LogicError") << " File not found " << theMaterialFileName;
  
  float sup,sen,cab,col,ele,oth,air;
  sup=sen=cab=col=ele=0.;

  std::string materialName;

  while(theMaterialFile) {
    theMaterialFile >> materialName;
    theMaterialFile >> sup >> sen >> cab >> col >> ele;

    if (materialName[0] == '#') //Ignore comments
      continue;
 
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
    edm::LogInfo("MaterialBudget") 
      << "MaterialBudgetCategorizer: Material " << materialName << " filled: " 
      << "\n\tSUP " << sup 
      << "\n\tSEN " << sen 
      << "\n\tCAB " << cab 
      << "\n\tCOL " << col 
      << "\n\tELE " << ele 
      << "\n\tOTH " << oth 
      << "\n\tAIR " << air
      << "\n\tAdd up to: " << sup + sen + cab + col + ele + oth + air;
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
