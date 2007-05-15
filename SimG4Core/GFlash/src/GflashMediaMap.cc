#include "SimG4Core/GFlash/interface/GflashMediaMap.h"
#include "SimG4Core/GFlash/interface/GflashCalorimeterNumber.h"

//#include "SimG4Core/Geometry/interface/G4LogicalVolumeToDDLogicalPartMapper.h"
//#include "DetectorDescription/Core/interface/DDName.h"
//#include "DetectorDescription/Core/interface/DDMaterial.h"
//#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "G4Material.hh"
#include "G4Element.hh"

#include "G4LogicalVolumeStore.hh"

GflashMediaMap::GflashMediaMap(G4Region* envelope) 
{ 
  buildGflashMediaMap(envelope);
  std::cout <<" GflashMediaMap::GflashMediaMap ... Printing MediaMap ..." << std::endl;
  printGflashMediaMap();
}

GflashMediaMap::~GflashMediaMap()
{ 
}

void GflashMediaMap::buildGflashMediaMap(G4Region* envelope)
{
  std::vector<G4LogicalVolume*>::iterator rootLVItr = 
    envelope->GetRootLogicalVolumeIterator();
  size_t nRootLV = envelope->GetNumberOfRootVolumes();
  
  if (!gIndexMap.empty()) gIndexMap.clear();
  if (!gMaterialMap.empty()) gMaterialMap.clear();
  
  for(size_t iLV=0 ; iLV < nRootLV ; iLV++)      {
    //Get the logical volumes in that region
    G4LogicalVolume* aLV = *rootLVItr;

    if (aLV->GetName() == "ESPM") {
      gIndexMap.insert(GflashIndexMap::value_type(aLV->GetName(),kESPM));
      gMaterialMap.insert(GflashMaterialMap::value_type(kESPM,aLV->GetMaterial()));
    }
    else if (aLV->GetName() == "ENCA") {
      gIndexMap.insert(GflashIndexMap::value_type(aLV->GetName(),kENCA));
      gMaterialMap.insert(GflashMaterialMap::value_type(kENCA,aLV->GetMaterial()));
    }
    else if (aLV->GetName() == "HB") {
      gIndexMap.insert(GflashIndexMap::value_type(aLV->GetName(),kHB));
      gMaterialMap.insert(GflashMaterialMap::value_type(kHB,aLV->GetMaterial()));
    }
    else if (aLV->GetName() == "HE") {
      gIndexMap.insert(GflashIndexMap::value_type(aLV->GetName(),kHE));
      gMaterialMap.insert(GflashMaterialMap::value_type(kHE,aLV->GetMaterial()));
    }

    rootLVItr++;
  }
}

void GflashMediaMap::printGflashMediaMap()
{
  G4cout << " ... Printing GflashMediaMap" << G4endl;;
  
  if (!gIndexMap.empty()) {
    G4cout << " GflashMediaMap::IndexMap <";
    for(GflashIndexMap::const_iterator iditer = gIndexMap.begin() ; iditer != gIndexMap.end() ; ++iditer)
      G4cout << "(" << iditer->first << ":" << iditer->second  << ")"; G4cout << ">"<< G4endl;
  }
  else {G4cout << " empty IndexMap! " << G4endl; }
  
  if (!gMaterialMap.empty()) {
    G4cout << " GflashMediaMap::MaterialMap <";
    for(GflashMaterialMap::const_iterator miter = gMaterialMap.begin() ; miter != gMaterialMap.end() ; ++miter)
      G4cout << "(" << miter->first << ":" << miter->second->GetName() << ")"; G4cout << ">" << G4endl;
  }
  else {G4cout << " empty MaterialMap! " << G4endl; }

  if (!gMaterialMap.empty()) {
    G4cout << " GflashMediaMap::MaterialMap <";
    for(GflashMaterialMap::const_iterator miter = gMaterialMap.begin() ; miter != gMaterialMap.end() ; ++miter)
      G4cout << "(" << miter->first << ":" << miter->second->GetDensity()/(g/cm3) << ")"; G4cout << ">" << G4endl;
  }
  else {G4cout << " empty MaterialMap! " << G4endl; }

} 

GflashCalorimeterNumber GflashMediaMap::getIndex(G4String name)
{
  GflashCalorimeterNumber index = kNULL;

  GflashIndexMap::const_iterator iter;
  
  iter = gIndexMap.find(name);

  if(iter != gIndexMap.end()) {
    index = iter->second;
  }
  else {
    G4cout << " GflashMediaMap::getIndex : Empty IndexMap " << G4endl;
  }

  return index;
}

G4Material* GflashMediaMap::getMaterial(GflashCalorimeterNumber kCalorimeter)
{
  G4Material* material = 0;

  GflashMaterialMap::const_iterator iter;
  iter = gMaterialMap.find(kCalorimeter);

  if(iter != gMaterialMap.end()) {
    material = iter->second;
  }
  else {
    G4cout << " GflashMediaMap::getMaterialMap : Empty MaterialMap " << G4endl;
  }

  return material;
}

GflashCalorimeterNumber GflashMediaMap::getCalorimeterNumber(const G4FastTrack& fastTrack)
{
  GflashCalorimeterNumber index =kNULL;
  G4String alogicalVolumeName = fastTrack.GetEnvelopeLogicalVolume()->GetName();
  index = getIndex(alogicalVolumeName);

  return index;
}
