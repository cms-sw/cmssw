#ifndef SimG4Core_Notification_Dispatcher_H
#define SimG4Core_Notification_Dispatcher_H

#include "sigc++/signal.h"
#include "sigc++/bind.h"

template<typename Event>
class Dispatcher : public sigc::signal<void(Event const *)> 
{
 public:
    typedef  sigc::signal<void(Event const *)> super;
    Dispatcher(Event const * ev) 
    { 
	m_conn = this->connect(sigc::bind(&super::operator(),this,_1)); 
	std::cout << " Dispatcher constructed " << std::endl;
    }
    ~Dispatcher() 
    { 
	m_conn.disconnect();
	std::cout << " Dispatcher destructed " << std::endl;
    }
    sigc::connection m_conn;
};
   
#endif


