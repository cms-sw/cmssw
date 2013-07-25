#ifndef HelpfulWatchers_SimTracer_h
#define HelpfulWatchers_SimTracer_h
// -*- C++ -*-
//
// Package:     HelpfulWatchers
// Class  :     SimTracer
// 
/**\class SimTracer SimTracer.h SimG4Core/HelpfulWatchers/interface/SimTracer.h

 Description: Prints a message for each Oscar signal

 Usage:
    <usage>

*/
//
// Original Author:  
//         Created:  Tue Nov 22 16:41:33 EST 2005
// $Id: SimTracer.h,v 1.2 2005/12/08 21:37:49 chrjones Exp $
//

// system include files
#include <iostream>

// user include files
#include "SimG4Core/Notification/interface/Observer.h"
#include "SimG4Core/Watcher/interface/SimWatcher.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "G4Step.hh"

// forward declarations
class DDDWorld;
class BeginOfJob;
class BeginOfRun;
class BeginOfEvent;
class BeginOfTrack;
class G4Step;

class EndOfRun;
class EndOfEvent;
class EndOfTrack;

#define OBSERVES(type) public Observer<const type*>
#define UPDATE(type) void update(const type*) { std::cout <<"++ signal " #type<<std::endl; }
class SimTracer : public SimWatcher, 
OBSERVES(DDDWorld),
OBSERVES(BeginOfJob),
OBSERVES(BeginOfRun),
OBSERVES(BeginOfEvent),
OBSERVES(BeginOfTrack),
OBSERVES(G4Step),
OBSERVES(EndOfRun),
OBSERVES(EndOfEvent),
OBSERVES(EndOfTrack)
{

   public:
   SimTracer(const edm::ParameterSet& pSet) : 
   m_verbose(pSet.getUntrackedParameter<bool>("verbose",false)) {
   }
     //virtual ~SimTracer();

      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
UPDATE(DDDWorld)
UPDATE(BeginOfJob)
UPDATE(BeginOfRun)
UPDATE(BeginOfEvent)
UPDATE(BeginOfTrack)
   void update(const G4Step* iStep) { 
   std::cout <<"++ signal G4Step " ;
   if(m_verbose) {
      const G4StepPoint* post = iStep->GetPostStepPoint();
      const G4ThreeVector pos = post->GetPosition();
      std::cout << "( "<<pos.x()<<","<<pos.y()<<","<<pos.z()<<") ";
      if(post->GetPhysicalVolume()) {
	 std::cout << post->GetPhysicalVolume()->GetName();
      }
   }
   std::cout <<std::endl; 
}
//UPDATE(G4Step)
UPDATE(EndOfRun)
UPDATE(EndOfEvent)
UPDATE(EndOfTrack)

   private:
     //SimTracer(const SimTracer&); // stop default

     //const SimTracer& operator=(const SimTracer&); // stop default

     // ---------- member data --------------------------------
     bool m_verbose;
};


#endif
