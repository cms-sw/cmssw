// -*- C++ -*-
//
// Package:     Notification
// Class  :     SimActivityRegistry
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Sun Nov 13 12:44:58 EST 2005
// $Id: SimActivityRegistry.cc,v 1.2 2005/11/21 22:01:21 chrjones Exp $
//

// system include files

// user include files
#include "SimG4Core/Notification/interface/SimActivityRegistry.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
//SimActivityRegistry::SimActivityRegistry()
//{
//}

// SimActivityRegistry::SimActivityRegistry(const SimActivityRegistry& rhs)
// {
//    // do actual copying here;
// }

//SimActivityRegistry::~SimActivityRegistry()
//{
//}

//
// assignment operators
//
// const SimActivityRegistry& SimActivityRegistry::operator=(const SimActivityRegistry& rhs)
// {
//   //An exception safe implementation is
//   SimActivityRegistry temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void
SimActivityRegistry::connect(SimActivityRegistry& iOther)
{
   beginOfJobSignal_.connect(iOther.beginOfJobSignal_);
   dddWorldSignal_.connect(iOther.dddWorldSignal_);
   beginOfRunSignal_.connect(iOther.beginOfRunSignal_);
   beginOfEventSignal_.connect(iOther.beginOfEventSignal_);
   beginOfTrackSignal_.connect(iOther.beginOfTrackSignal_);
   g4StepSignal_.connect(iOther.g4StepSignal_);

   endOfRunSignal_.connect(iOther.endOfRunSignal_);
   endOfEventSignal_.connect(iOther.endOfEventSignal_);
   endOfTrackSignal_.connect(iOther.endOfTrackSignal_);
}
//
// const member functions
//

//
// static member functions
//
