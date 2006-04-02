#include "Validation/Geometry/interface/MaterialBudgetCategorizer.h"
//#include "Utilities/UI/interface/SimpleConfigurable.h"

#include "G4LogicalVolumeStore.hh"
#include "G4Material.hh"

#include<fstream.h>
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
    theMaterialMap[ (*matTable)[ii]->GetName()] = ii++;
  }

}
