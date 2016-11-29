#ifndef Watcher_SimWatcherMaker_h
#define Watcher_SimWatcherMaker_h
// -*- C++ -*-
//
// Package:     Watcher
// Class  :     SimWatcherMaker
// 
/**\class SimWatcherMaker SimWatcherMaker.h SimG4Core/Watcher/interface/SimWatcherMaker.h

 Description: Makes a particular type of SimWatcher

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
#include "SimG4Core/Watcher/interface/SimWatcherMakerBase.h"
#include "SimG4Core/Notification/interface/SimActivityRegistryEnroller.h"

// forward declarations

template<class T>
class SimWatcherMaker : public SimWatcherMakerBase
{

   public:
      SimWatcherMaker(){}

      // ---------- const member functions ---------------------
      virtual void make(const edm::ParameterSet& p,
			SimActivityRegistry& reg,
			std::shared_ptr<SimWatcher>& oWatcher,
			std::shared_ptr<SimProducer>& oProd
	 ) const
      {
	auto returnValue = std::make_shared<T>(p);
	SimActivityRegistryEnroller::enroll(reg, returnValue.get());
	oWatcher = returnValue;

	//If this is also a SimProducer, set the value
	oProd = this->getSimProducer(returnValue.get(), returnValue);
      }

   private:
      std::shared_ptr<SimProducer>
      getSimProducer(SimProducer*, std::shared_ptr<T>& iProd) const{
	 return std::shared_ptr<SimProducer>(iProd);
      }
      std::shared_ptr<SimProducer>
      getSimProducer(void*, std::shared_ptr<T>& iProd) const{
	 return std::shared_ptr<SimProducer>();
      }

};


#endif
