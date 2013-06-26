#ifndef Watcher_SimWatcherMakerBase_h
#define Watcher_SimWatcherMakerBase_h
// -*- C++ -*-
//
// Package:     Watcher
// Class  :     SimWatcherMakerBase
// 
/**\class SimWatcherMakerBase SimWatcherMakerBase.h SimG4Core/Watcher/interface/SimWatcherMakerBase.h

 Description: Base class for the 'maker' which creates Watchers

 Usage:
    This class is the interface for creating a Watcher and for connnecting
 the appropriate OSCAR signals to that Watcher

*/
//
// Original Author:  Chris D Jones
//         Created:  Tue Nov 22 13:03:39 EST 2005
// $Id: SimWatcherMakerBase.h,v 1.2 2005/11/29 18:40:25 chrjones Exp $
//

// system include files
#include "boost/shared_ptr.hpp"

// user include files

// forward declarations
class SimActivityRegistry;
namespace edm{
  class ParameterSet;
}
class SimWatcher;
class SimProducer;
class SimWatcherMakerBase
{

   public:
      SimWatcherMakerBase() {}
      virtual ~SimWatcherMakerBase() {}

      // ---------- const member functions ---------------------
      virtual void make(const edm::ParameterSet&,
			SimActivityRegistry&,
			boost::shared_ptr<SimWatcher>&,
			boost::shared_ptr<SimProducer>&
	 ) const = 0;
};


#endif
