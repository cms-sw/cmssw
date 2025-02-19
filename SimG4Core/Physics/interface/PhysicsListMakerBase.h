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
// $Id: PhysicsListMakerBase.h,v 1.4 2010/07/29 22:40:22 sunanda Exp $
//

// system include files
#include <memory>

// user include files
#include "HepPDT/ParticleDataTable.hh"

// forward declarations
class SimActivityRegistry;
namespace edm{
  class ParameterSet;
}
namespace sim {
   class FieldBuilder;
}

class PhysicsListMakerBase
{

   public:
      PhysicsListMakerBase() {}
      virtual ~PhysicsListMakerBase() {}

      // ---------- const member functions ---------------------
      virtual std::auto_ptr<PhysicsList> make(G4LogicalVolumeToDDLogicalPartMap&,
					      const HepPDT::ParticleDataTable * ,
					      sim::FieldBuilder *,
					      const edm::ParameterSet&,
					      SimActivityRegistry&) const = 0;

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------

   private:
      //PhysicsListMakerBase(const PhysicsListMakerBase&); // stop default

      //const PhysicsListMakerBase& operator=(const PhysicsListMakerBase&); // stop default

      // ---------- member data --------------------------------

};


#endif
