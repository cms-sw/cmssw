#ifndef Watcher_SimWatcherMaker_h
#define Watcher_SimWatcherMaker_h
// -*- C++ -*-
//
// Package:     Watcher
// Class  :     SimWatcherMaker
// 
/**\class SimWatcherMaker SimWatcherMaker.h SimG4Core/Watcher/interface/SimWatcherMaker.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  
//         Created:  Tue Nov 22 13:03:44 EST 2005
// $Id: SimWatcherMaker.h,v 1.1 2005/11/22 20:05:22 chrjones Exp $
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
      virtual std::auto_ptr<SimWatcher> make(const edm::ParameterSet& p,
                                SimActivityRegistry& reg) const
      {
	std::auto_ptr<T> returnValue(new T(p));
	SimActivityRegistryEnroller::enroll(reg, returnValue.get());
	
	return std::auto_ptr<SimWatcher>(returnValue);
      }

};


#endif
