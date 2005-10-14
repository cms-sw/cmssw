#ifndef SimG4Core_Notification_Dispatcher_H
#define SimG4Core_Notification_Dispatcher_H

#include <boost/signal.hpp>
#include <boost/bind.hpp>
#include <boost/signals/connection.hpp>

template<typename Event>
class Dispatcher : public boost::signal<void(Event const *)> 
{
 public:
    typedef  boost::signal<void(Event const *)> super;
    Dispatcher(Event const * ev) 
    { 
	m_conn = this->connect(boost::bind(&super::operator(),this,_1)); 
	std::cout << " Dispatcher constructed " << std::endl;
    }
    ~Dispatcher() 
    { 
	m_conn.disconnect();
	std::cout << " Dispatcher destructed " << std::endl;
    }
    boost::signals::connection m_conn;
};
   
#endif


