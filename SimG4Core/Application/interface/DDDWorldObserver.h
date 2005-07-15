#ifndef SimG4Core_DDDWorldObserver_H
#define SimG4Core_DDDWorldObserver_H

#include "Utilities/Notification/interface/DispatcherObserver.h"
#include "SimG4Core/Geometry/interface/DDDWorld.h"

class DDDWorldObserver : private frappe::Observer<DDDWorld> 
{
public:
  typedef frappe::Observer<DDDWorld> super;
  DDDWorldObserver(seal::Context* ic, const std::string& iname) :super(ic) 
  { std::cout << " DDDWorldObserver constructed " << std::endl; initialize(); }
private:
  virtual void update(const DDDWorld * w) 
  { std::cout << " in update " << std::endl; if (w!=0) std::cout << " got DDDWorld " << std::endl; }
};

#endif
