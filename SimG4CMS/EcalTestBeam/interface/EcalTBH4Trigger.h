#ifndef HelpfulWatchers_EcalTBH4Trigger_h
#define HelpfulWatchers_EcalTBH4Trigger_h
// -*- C++ -*-
//
// Package:     HelpfulWatchers
// Class  :     EcalTBH4Trigger
// 
/**\class EcalTBH4Trigger EcalTBH4Trigger.h SimG4Core/HelpfulWatchers/interface/EcalTBH4Trigger.h

 Description: Simulates ECALTBH4 trigger an throw exception in case of non triggering event

 Usage:
    <usage>

*/
// $Id: EcalTBH4Trigger.h,v 1.1 2007/03/19 17:21:49 fabiocos Exp $
//

// system include files
#include <iostream>

// user include files
#include "SimG4Core/Notification/interface/Observer.h"
#include "SimG4Core/Watcher/interface/SimWatcher.h"
#include "SimG4Core/Notification/interface/SimG4Exception.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "G4Step.hh"
#include "G4VProcess.hh"
#include "G4VTouchable.hh"

// forward declarations
class DDDWorld;
class BeginOfJob;
class BeginOfRun;
class BeginOfEvent;
class BeginOfTrack;

class EndOfRun;
class EndOfEvent;
class EndOfTrack;

#define OBSERVES(type) public Observer<const type*>
#define UPDATEH4(type) void update(const type*) { }
class EcalTBH4Trigger : public SimWatcher, 
			//OBSERVES(DDDWorld),
			//OBSERVES(BeginOfJob),
			//OBSERVES(BeginOfRun),
			OBSERVES(BeginOfEvent),
			//OBSERVES(BeginOfTrack),
			OBSERVES(G4Step),
			//OBSERVES(EndOfRun),
			OBSERVES(EndOfEvent)
     //,
     //OBSERVES(EndOfTrack)
{

 public:
  EcalTBH4Trigger(const edm::ParameterSet& pSet) : 
    m_verbose(pSet.getUntrackedParameter<bool>("verbose",false)), nTriggeredEvents_(0), trigEvents_(pSet.getUntrackedParameter<int>("trigEvents",-1)) {
  }
  //virtual ~EcalTBH4Trigger();
  
  // ---------- const member functions ---------------------

  // ---------- static member functions --------------------
  
  // ---------- member functions ---------------------------
  //  UPDATEH4(DDDWorld)
  //UPDATEH4(BeginOfJob)
  //UPDATEH4(BeginOfRun)
    //  UPDATEH4(BeginOfEvent)
  void update(const BeginOfEvent* anEvent) 
    {
      //      std::cout <<"++ signal BeginOfEvent " ;
      //      m_enteringTBH4BeamLine=false;
      //      m_exitingTBH4BeamLine=false;
      //      m_passedTrigger=false;
      m_passedTrg1=false;
      m_passedTrg3=false;
      m_passedTrg4=false;
      m_passedTrg5=false;
      m_passedTrg6=false;
    } 

  //  UPDATEH4(BeginOfTrack)
  void update(const G4Step* iStep) 
    { 
      if (trigEvents_ >= 0 && nTriggeredEvents_ >= trigEvents_)
	throw SimG4Exception("Number of needed trigger events reached in ECALTBH4");

      const G4StepPoint* pre = iStep->GetPreStepPoint(); 
      const G4StepPoint* post = iStep->GetPostStepPoint(); 
      if(m_verbose) {
	std::cout <<"++ signal G4Step" ;
	const G4VTouchable* touch = iStep->GetPreStepPoint()->GetTouchable();
	//Get name and copy numbers
	if (touch->GetHistoryDepth() > 0) {
	  for (int ii = 0; ii <= touch->GetHistoryDepth() ; ii++) {
	    std::cout << "EcalTBH4::Level " << ii
		      << ": " << touch->GetVolume(ii)->GetName() << "["
		      << touch->GetReplicaNumber(ii) << "]";
	  }
	}
	std::cout <<std::endl; 
	const G4Track* theTrack = iStep->GetTrack();
	const G4ThreeVector pos = post->GetPosition();
	std::cout << "( "<<pos.x()<<","<<pos.y()<<","<<pos.z()<<") ";
	std::cout << " released energy (MeV) " <<  iStep->GetTotalEnergyDeposit()/MeV  ;
	if (theTrack)
	  {
	    const G4ThreeVector mom = theTrack->GetMomentum();
	    std::cout << " track length (cm) " << theTrack->GetTrackLength()/cm
		      << " particle type " << theTrack->GetDefinition()->GetParticleName()
		      << " momentum " << "( "<<mom.x()<<","<<mom.y()<<","<<mom.z()<<") ";
	    if (theTrack->GetCreatorProcess())
	      std::cout << " created by " << theTrack->GetCreatorProcess()->GetProcessName();
	  }
	if(post->GetPhysicalVolume()) {
	  std::cout << " " << pre->GetPhysicalVolume()->GetName() << "->" << post->GetPhysicalVolume()->GetName();
	}
	std::cout <<std::endl; 
      }
      
      if (post && post->GetPhysicalVolume())
	{
	  
	  if (!m_passedTrg1 && post->GetPhysicalVolume()->GetName() == "TRG1")
	    m_passedTrg1 = true;
	  if (!m_passedTrg3 && post->GetPhysicalVolume()->GetName() == "TRG3")
	    m_passedTrg3 = true;
	  if (!m_passedTrg4 && post->GetPhysicalVolume()->GetName() == "TRG4")
	    m_passedTrg4 = true;
	  if (!m_passedTrg5 && post->GetPhysicalVolume()->GetName() == "TRG5")
	    m_passedTrg5 = true;
	  if (!m_passedTrg6 && post->GetPhysicalVolume()->GetName() == "TRG6")
	    m_passedTrg6 = true;
	  if (post->GetPhysicalVolume()->GetName() == "CMSSE") //Exiting TBH4BeamLine
	    if (! (m_passedTrg1 && m_passedTrg6) ) // Trigger defined as Trg4 && Trg6
	      throw SimG4Exception("Event is not triggered by ECALTBH4");
      }
    
/*     if (!m_enteringTBH4BeamLine && ( post->GetPhysicalVolume()->GetName() ==  */

}
//UPDATEH4(G4Step)
//UPDATEH4(EndOfRun)
//UPDATEH4(EndOfEvent)
  void update(const EndOfEvent* anEvent) 
    {
      //      std::cout <<"++ signal BeginOfEvent " ;
      //      m_enteringTBH4BeamLine=false;
      //      m_exitingTBH4BeamLine=false;
      //      m_passedTrigger=false;
      nTriggeredEvents_++;
    } 
//UPDATEH4(EndOfTrack)

   private:
     //EcalTBH4Trigger(const EcalTBH4Trigger&); // stop default

     //const EcalTBH4Trigger& operator=(const EcalTBH4Trigger&); // stop default

     // ---------- member data --------------------------------
 bool m_verbose;
// bool m_enteringTBH4BeamLine; 
// bool m_exitingTBH4BeamLine; 
 bool m_passedTrg1;
 bool m_passedTrg3;
 bool m_passedTrg4;
 bool m_passedTrg5;
 bool m_passedTrg6;
 int nTriggeredEvents_;
 int trigEvents_;
 // bool m_passedTrigger; 
};


#endif
