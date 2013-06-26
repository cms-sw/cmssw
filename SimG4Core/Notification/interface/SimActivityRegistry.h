#ifndef SimG4Core_Notification_SimActivityRegistry_h
#define SimG4Core_Notification_SimActivityRegistry_h
// -*- C++ -*-
//
// Package:     Notification
// Class  :     SimActivityRegistry
// 
/**\class SimActivityRegistry SimActivityRegistry.h SimG4Core/Notification/interface/SimActivityRegistry.h

 Description: Holds the various signals emitted in the simulation framework

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Sun Nov 13 11:43:40 EST 2005
// $Id: SimActivityRegistry.h,v 1.7 2007/12/02 05:17:47 chrjones Exp $
//

// system include files
#include "boost/bind.hpp"
#include "boost/mem_fn.hpp"

// user include files
#include "SimG4Core/Notification/interface/Signaler.h"


// forward declarations
class BeginOfJob;
class BeginOfRun;
class BeginOfEvent;
class BeginOfTrack;
class BeginOfStep;
class EndOfRun;
class EndOfEvent;
class EndOfTrack;
class DDDWorld;
class G4Step;

#define SAR_CONNECT_METHOD(signal) void connect(Observer<const signal*>* iObject) { watch ## signal (iObject); }

class SimActivityRegistry
{

   public:
      SimActivityRegistry() {}
      //virtual ~SimActivityRegistry();

      typedef sim_act::Signaler<BeginOfJob> BeginOfJobSignal;
      BeginOfJobSignal beginOfJobSignal_;
      void watchBeginOfJob(const BeginOfJobSignal::slot_type& iSlot){
         beginOfJobSignal_.connect(iSlot);
      }
      SAR_CONNECT_METHOD(BeginOfJob)

      typedef sim_act::Signaler<DDDWorld> DDDWorldSignal;
      DDDWorldSignal dddWorldSignal_;
      void watchDDDWorld(const DDDWorldSignal::slot_type& iSlot){
         dddWorldSignal_.connect(iSlot);
      }
      SAR_CONNECT_METHOD(DDDWorld)

      typedef sim_act::Signaler<BeginOfRun> BeginOfRunSignal;
      BeginOfRunSignal beginOfRunSignal_;
      void watchBeginOfRun(const BeginOfRunSignal::slot_type& iSlot){
         beginOfRunSignal_.connect(iSlot);
      }
      SAR_CONNECT_METHOD(BeginOfRun)

      typedef sim_act::Signaler<BeginOfEvent> BeginOfEventSignal;
      BeginOfEventSignal beginOfEventSignal_;
      void watchBeginOfEvent(const BeginOfEventSignal::slot_type& iSlot){
         beginOfEventSignal_.connect(iSlot);
      }
      SAR_CONNECT_METHOD(BeginOfEvent)

      typedef sim_act::Signaler<BeginOfTrack> BeginOfTrackSignal;
      BeginOfTrackSignal beginOfTrackSignal_;
      void watchBeginOfTrack(const BeginOfTrackSignal::slot_type& iSlot){
         beginOfTrackSignal_.connect(iSlot);
      }
      SAR_CONNECT_METHOD(BeginOfTrack)
      
      typedef sim_act::Signaler<G4Step> G4StepSignal;
      G4StepSignal g4StepSignal_;
      void watchG4Step(const G4StepSignal::slot_type& iSlot){
         g4StepSignal_.connect(iSlot);
      }
      SAR_CONNECT_METHOD(G4Step)
         
      typedef sim_act::Signaler<EndOfRun> EndOfRunSignal;
      EndOfRunSignal endOfRunSignal_;
      void watchEndOfRun(const EndOfRunSignal::slot_type& iSlot){
         endOfRunSignal_.connect(iSlot);
      }
      SAR_CONNECT_METHOD(EndOfRun)
         
      typedef sim_act::Signaler<EndOfEvent> EndOfEventSignal;
      EndOfEventSignal endOfEventSignal_;
      void watchEndOfEvent(const EndOfEventSignal::slot_type& iSlot){
         endOfEventSignal_.connect(iSlot);
      }
      SAR_CONNECT_METHOD(EndOfEvent)
         
      typedef sim_act::Signaler<EndOfTrack> EndOfTrackSignal;
      EndOfTrackSignal endOfTrackSignal_;
      void watchEndOfTrack(const EndOfTrackSignal::slot_type& iSlot){
         endOfTrackSignal_.connect(iSlot);
      }
      SAR_CONNECT_METHOD(EndOfTrack)
         
      ///forwards our signals to slots connected to iOther
      void connect(SimActivityRegistry& iOther);
      
   private:
      SimActivityRegistry(const SimActivityRegistry&); // stop default

      const SimActivityRegistry& operator=(const SimActivityRegistry&); // stop default

      // ---------- member data --------------------------------

};


#endif
