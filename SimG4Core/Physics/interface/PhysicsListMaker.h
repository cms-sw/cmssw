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
      virtual std::unique_ptr<PhysicsList> make(G4LogicalVolumeToDDLogicalPartMap& map_,
					      const HepPDT::ParticleDataTable * table_,
					      sim::ChordFinderSetter * chordFinderSetter_,
					      const edm::ParameterSet& p,
					      SimActivityRegistry& reg) const
      {
	std::unique_ptr<T> returnValue(new T(map_, table_, chordFinderSetter_, p));
	SimActivityRegistryEnroller::enroll(reg, returnValue.get());
	
	return returnValue;
      }
   private:

      PhysicsListMaker(const PhysicsListMaker&) = delete;
      const PhysicsListMaker& operator=(const PhysicsListMaker&) = delete; 

};


#endif
