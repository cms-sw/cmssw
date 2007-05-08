#ifndef Physics_PhysicsListMaker_h
#define Physics_PhysicsListMaker_h
// -*- C++ -*-
//
// Package:     Physics
// Class  :     PhysicsListMaker
// 
/**\class PhysicsListMaker PhysicsListMaker.h SimG4Core/Physics/interface/PhysicsListMaker.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  
//         Created:  Tue Nov 22 13:03:44 EST 2005
// $Id: PhysicsListMaker.h,v 1.1 2005/11/22 20:05:22 chrjones Exp $
//

// system include files
#include <memory>

// user include files
#include "SimG4Core/Physics/interface/PhysicsListMakerBase.h"
#include "SimG4Core/Notification/interface/SimActivityRegistryEnroller.h"

// forward declarations

template<class T>
class PhysicsListMaker : public PhysicsListMakerBase
{

   public:
      PhysicsListMaker(){}

      // ---------- const member functions ---------------------
      virtual std::auto_ptr<PhysicsList> make(G4LogicalVolumeToDDLogicalPartMap& map_,
					      const edm::ParameterSet& p,
					      SimActivityRegistry& reg) const
      {
	std::auto_ptr<T> returnValue(new T(map_, p));
	SimActivityRegistryEnroller::enroll(reg, returnValue.get());
	
	return std::auto_ptr<PhysicsList>(returnValue);
      }

};


#endif
