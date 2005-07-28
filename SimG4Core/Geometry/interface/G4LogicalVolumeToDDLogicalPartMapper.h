#ifndef SimG4Core_G4LogicaVolumeToDDLogicalPartMapper_H
#define SimG4Core_G4LogicaVolumeToDDLogicalPartMapper_H

#include "DetectorDescription/Core/interface/DDMapper.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "SimG4Core/Notification/interface/Singleton.h"

#include "G4LogicalVolume.hh"

typedef Singleton<DDMapper<G4LogicalVolume*,DDLogicalPart> > G4LogicalVolumeToDDLogicalPartMapper;
typedef DDMapper<G4LogicalVolume*,DDLogicalPart> ConcreteG4LogicalVolumeToDDLogicalPartMapper;

#endif
