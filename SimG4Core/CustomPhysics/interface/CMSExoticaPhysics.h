#ifndef SimG4Core_CustomPhysics_CMSExoticaPhysics_H
#define SimG4Core_CustomPhysics_CMSExoticaPhysics_H
 
// 
//
// Author: V.Ivantchenko  7 May 2018
//     
// Description: general interface to exotic particle physics 
//

#include "SimG4Core/Physics/interface/PhysicsList.h"
 
class CMSExoticaPhysics{

public:
  CMSExoticaPhysics(PhysicsList* phys, const edm::ParameterSet & p);
  ~CMSExoticaPhysics() = default;
};
 
#endif
