#ifndef Physics_PhysicsListMakerBase_h
#define Physics_PhysicsListMakerBase_h
// -*- C++ -*-
//
// Package:     Physics
// Class  :     PhysicsListMakerBase
//
/**\class PhysicsListMakerBase PhysicsListMakerBase.h SimG4Core/Physics/interface/PhysicsListMakerBase.h

 Description: Base class for the 'maker' which creates PhysicsLists

 Usage:
    This class is the interface for creating a physics list and for connnecting
 the appropriate OSCAR signals to that physics list

*/
//
// Original Author:  Chris D Jones
//         Created:  Tue Nov 22 13:03:39 EST 2005
//

// system include files
#include <memory>

// user include files
#include "SimG4Core/Physics/interface/PhysicsList.h"

// forward declarations
class SimActivityRegistry;
namespace edm {
  class ParameterSet;
}

class PhysicsListMakerBase {
public:
  PhysicsListMakerBase() {}
  virtual ~PhysicsListMakerBase() {}

  virtual std::unique_ptr<PhysicsList> make(const edm::ParameterSet&, SimActivityRegistry&) const = 0;

  PhysicsListMakerBase(const PhysicsListMakerBase&) = delete;
  const PhysicsListMakerBase& operator=(const PhysicsListMakerBase&) = delete;
};

#endif
