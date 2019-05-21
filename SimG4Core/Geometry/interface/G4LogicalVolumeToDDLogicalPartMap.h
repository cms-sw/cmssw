#ifndef SimG4Core_G4LogicaVolumeToDDLogicalPartMap_H
#define SimG4Core_G4LogicaVolumeToDDLogicalPartMap_H

#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDMapper.h"

#include "G4LogicalVolume.hh"

typedef DDMapper<G4LogicalVolume *, DDLogicalPart> G4LogicalVolumeToDDLogicalPartMap;

#endif
