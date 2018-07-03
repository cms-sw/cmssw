#ifndef SimG4Core_PhysicsLists_FTFCMS_BIC_H
#define SimG4Core_PhysicsLists_FTFCMS_BIC_H

#include "SimG4Core/Physics/interface/PhysicsList.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class FTFCMS_BIC: public PhysicsList {

public:
  FTFCMS_BIC(const edm::ParameterSet & p);
};

#endif



