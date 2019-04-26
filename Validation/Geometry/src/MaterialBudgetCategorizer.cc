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
  float air,cables,copper,h_scintillator,lead,hgc_g10_fr4,silicon,stainlesssteel,wcu,oth,epoxy,kapton,aluminium; 
  air=cables=copper=h_scintillator=lead=hgc_g10_fr4=silicon=stainlesssteel=wcu=epoxy=kapton=aluminium=0.;

  std::string materialName;
  while(theMaterialFile) {
    theMaterialFile >> materialName;
    theMaterialFile >> air >> cables >> copper >> h_scintillator >> lead >> hgc_g10_fr4 >> silicon >> stainlesssteel >> wcu >> epoxy >> kapton >> aluminium;
    // Skip comments
    if (materialName[0] == '#')
      continue;
    // Substitute '*' with spaces
    std::replace(materialName.begin(), materialName.end(), '*', ' ');
    oth = 0.000;
    theMap[materialName].clear();        // clear before re-filling
    theMap[materialName].push_back(air             ); // air
    theMap[materialName].push_back(cables          ); // cables          
    theMap[materialName].push_back(copper          ); // copper          
    theMap[materialName].push_back(h_scintillator  ); // h_scintillator  
    theMap[materialName].push_back(lead            ); // lead            
    theMap[materialName].push_back(hgc_g10_fr4); // hgc_g10_fr4
    theMap[materialName].push_back(silicon         ); // silicon         
    theMap[materialName].push_back(stainlesssteel  ); // stainlesssteel
    theMap[materialName].push_back(wcu             ); // wcu
    theMap[materialName].push_back(oth             ); // oth
    theMap[materialName].push_back(epoxy             ); // epoxy
    theMap[materialName].push_back(kapton             ); // kapton
    theMap[materialName].push_back(aluminium             ); // aluminium
    edm::LogInfo("MaterialBudget") 
      << "MaterialBudgetCategorizer: material " << materialName << " filled " 
      << std::endl
      << "\tair              " << air << std::endl
      << "\tcables           " << cables << std::endl
      << "\tcopper           " << copper << std::endl
      << "\th_scintillator   " << h_scintillator << std::endl
      << "\tlead             " << lead << std::endl
      << "\thgc_g10_fr4      " << hgc_g10_fr4 << std::endl
      << "\tsilicon          " << silicon << std::endl
      << "\tstainlesssteel   " << stainlesssteel << std::endl
      << "\twcu              " << wcu << std::endl
      << "\tepoxy              " << epoxy << std::endl
      << "\tkapton            " << kapton<< std::endl
      << "\taluminium            " << aluminium<< std::endl
      << "\tOTH              " << oth;
  }

}
