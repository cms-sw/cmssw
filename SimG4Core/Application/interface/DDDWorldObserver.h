#ifndef SimG4Core_DDDWorldObserver_H
#define SimG4Core_DDDWorldObserver_H

#include "SimG4Core/Notification/interface/Observer.h"
#include "SimG4Core/Geometry/interface/DDDWorld.h"

#include <iostream>

template<typename Event>
class DDDWorldObserver : public Observer<const Event*>
{
public:
    virtual void update(const DDDWorld * w)
    {
	std::cout << " DDDWorldObserver: in update " << std::endl;
	if (w!=0)
        std::cout << " DDDWorldObserver: got DDDWorld! " << std::endl;	
    }
};

#endif
