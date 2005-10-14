#ifndef SimG4Core_DDDWorldObserver_H
#define SimG4Core_DDDWorldObserver_H

#include "SimG4Core/Notification/interface/Observer.h"
#include "SimG4Core/Geometry/interface/DDDWorld.h"

#include <iostream>

template<typename Event>
class DDDWorldObserver : public Observer<Event>
{
public:
    virtual void init() 
    {
	std::cout << " DDDWorldObserver: in init " << std::endl;
	std::cout << " DDDWorldObserver: constructed " << std::endl; 
    }
    virtual void end() 
    {
	std::cout << " DDDWorldObserver: in end " << std::endl;
	std::cout << " DDDWorldObserver: destructed " << std::endl;
    }
    virtual void update(const DDDWorld * w) const
    {
	std::cout << " DDDWorldObserver: in update " << std::endl;
	if (w!=0)
        std::cout << " DDDWorldObserver: got DDDWorld! " << std::endl;	
    }
};

#endif
