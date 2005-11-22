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
// $Id: SimWatcherMakerBase.h,v 1.1 2005/11/22 20:05:22 chrjones Exp $
//

// system include files
#include <memory>

// user include files

// forward declarations
class SimActivityRegistry;
namespace edm{
  class ParameterSet;
}
class SimWatcher;

class SimWatcherMakerBase
{

   public:
      SimWatcherMakerBase() {}
      virtual ~SimWatcherMakerBase() {}

      // ---------- const member functions ---------------------
      virtual std::auto_ptr<SimWatcher> make(const edm::ParameterSet&,
					      SimActivityRegistry&) const = 0;


   private:

};


#endif
