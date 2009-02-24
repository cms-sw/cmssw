#ifndef HelpfulWatchers_G4StepStatistics_h
#define HelpfulWatchers_G4StepStatistics_h
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
#include "G4VProcess.hh"
#include "G4ParticleDefinition.hh"
#include <map>

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

//Define a class MyStepID
class MyStepID {
 private:
  //G4 Physical Volume
  //G4String MyG4PhysicalVolume;
  //G4 Region
  G4String MyG4RegionName;
  //G4 Physical Process
  G4String MyG4ProcessName;
  //Particle PDG ID
  G4int MyParticlePDGID;
  //Particle Name
  //G4String MyParticleName;
 public:
  MyStepID(const G4Step* MyG4Step)
    :
    //MyG4PhysicalVolume("UNDEFINED"),
    MyG4RegionName("UNDEFINED"),
    MyG4ProcessName("UNDEFINED"),
    MyParticlePDGID(MyG4Step->GetTrack()->GetDefinition()->GetPDGEncoding())
    //MyParticleName(MyG4Step->GetTrack()->GetDefinition()->GetParticleName())
    {
      if (MyG4Step->GetPostStepPoint()->GetPhysicalVolume())
	{
	  //MyG4PhysicalVolume = MyG4Step->GetPostStepPoint()->GetPhysicalVolume()->GetName();
	  MyG4RegionName = MyG4Step->GetPostStepPoint()->GetPhysicalVolume()->GetLogicalVolume()->GetRegion()->GetName();
	}
      if (MyG4Step->GetTrack()->GetCreatorProcess())
	{
	  MyG4ProcessName = MyG4Step->GetTrack()->GetCreatorProcess()->GetProcessName();
	}
    }
  //Default Constructor:
  MyStepID()
    :
    //MyG4PhysicalVolume("UNDEFINED"),
    MyG4RegionName("UNDEFINED"),
    MyG4ProcessName("UNDEFINED"),
    MyParticlePDGID(-9999999)
    //MyParticleName(MyG4Step->GetTrack()->GetDefinition()->GetParticleName())
    {}
  //Getters
  //G4String GetPhysicalVolumeName () const { return MyG4PhysicalVolume; }
  G4String GetRegionName () const { return MyG4RegionName; }
  G4String GetProcessName () const { return MyG4ProcessName; }
  G4int GetParticlePDGID () const { return MyParticlePDGID; }
  //G4String GetParticleName () const { return MyParticleName; } 
  //Comparison Operators (necessary in order to use StepID as a key in a map)
  bool operator==(const MyStepID& id) const 
    {
      //std::cout<<"Operator == (MyStepID)"<<std::endl;
      //std::cout<<MyG4ProcessName<<" "<<id.GetRegionName()<<
      //MyG4ProcessName<<" "<<id.GetProcessName()<<
      //MyParticlePDGID<<" "<<id.GetParticlePDGID()<<std::endl;
      return (MyG4RegionName==id.GetRegionName() && MyG4ProcessName==id.GetProcessName() && MyParticlePDGID==id.GetParticlePDGID());
    }
  //bool operator!=(const MyStepID& id) const 
  //  {
      //std::cout<<"Operator != (MyStepID)"<<std::endl;
      //std::cout<<MyG4ProcessName<<" "<<id.GetRegionName()<<
      //MyG4ProcessName<<" "<<id.GetProcessName()<<
      //MyParticlePDGID<<" "<<id.GetParticlePDGID()<<std::endl;
  //    return (MyG4RegionName!=id.GetRegionName() || MyG4ProcessName!=id.GetProcessName() || MyParticlePDGID!=id.GetParticlePDGID());
  //  }
  bool operator<(const MyStepID& id) const 
    {
      //std::cout<<"Operator < (MyStepID)"<<std::endl;
      //std::cout<<MyG4ProcessName<<" "<<id.GetRegionName()<<
      //MyG4ProcessName<<" "<<id.GetProcessName()<<
      //MyParticlePDGID<<" "<<id.GetParticlePDGID()<<std::endl;
      if (MyParticlePDGID != id.GetParticlePDGID()){
	return (MyParticlePDGID > id.GetParticlePDGID());
      }
      else if (MyG4RegionName != id.GetRegionName()){
	return (MyG4RegionName > id.GetRegionName());
      }
      else if (MyG4ProcessName != id.GetProcessName()){
	return (MyG4ProcessName > id.GetProcessName());
      }
    }
  bool operator>(const MyStepID& id) const 
    {
      //std::cout<<"Operator > (MyStepID)"<<std::endl;
      //std::cout<<MyG4ProcessName<<" "<<id.GetRegionName()<<
      //MyG4ProcessName<<" "<<id.GetProcessName()<<
      //MyParticlePDGID<<" "<<id.GetParticlePDGID()<<std::endl;
      if (MyParticlePDGID != id.GetParticlePDGID()){
	return (MyParticlePDGID < id.GetParticlePDGID());
      }
      else if (MyG4RegionName != id.GetRegionName()){
	return (MyG4RegionName < id.GetRegionName());
      }
      else if (MyG4ProcessName != id.GetProcessName()){
	return (MyG4ProcessName < id.GetProcessName());
      }
    }

};


#define OBSERVES(type) public Observer<const type*>
#define UPDATE(type) void update(const type*) { std::cout <<"++ signal " #type<<std::endl; }
class G4StepStatistics : public SimWatcher, 
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
   G4StepStatistics(const edm::ParameterSet& pSet) : 
   m_verbose(pSet.getUntrackedParameter<bool>("verbose",false))  {
   }
UPDATE(DDDWorld)
UPDATE(BeginOfJob)
UPDATE(BeginOfRun)
  //    void update(const BeginOfRun* iRun) {
  //std::cout <<"++ signal BeginOfRun " <<std::endl;
  //}
UPDATE(BeginOfEvent)
UPDATE(BeginOfTrack)
   void update(const G4Step* iStep) { 
   std::cout <<"++ signal G4Step " ;
   //Dump the relevant information from the G4 step in the object mysteptest
   MyStepID mysteptest(iStep);
   //std::pair<MyStepID,int> pippo;
   std::map<MyStepID,int>::iterator check=G4StatsMap.find(mysteptest);
   //Add the StepID to the map, or increment if it already exists:
   if ( check == G4StatsMap.end() )
     {
       //pippo.first = mysteptest;
       //pippo.second = 1;
       //if (G4StatsMap.insert(std::make_pair(mysteptest,1))){
       // std::cout<<"ADDED STEP!"<<std::endl;
       //}
       G4StatsMap[mysteptest]=1;
	 //else {
	 //std::cout<<"FAILED TO ADD STEP!"<<std::endl;
	 //}
       std::cout << "Added to map StepID "
	 //		 <<mysteptest.GetPhysicalVolumeName()
	 	 <<" "
	 	 <<mysteptest.GetRegionName()
	 	 <<" "
	 	 <<mysteptest.GetProcessName()
	 	 <<" "
	 	 <<mysteptest.GetParticlePDGID()//;
		 <<" Number of Steps: "
		 <<G4StatsMap[mysteptest];
     }
   else
     {
       //G4StatsMap[mysteptest] += 1;
       //G4StatsMap.insert(std::make_pair(mysteptest,G4StatsMap[mysteptest]+1));
       //int c=G4StatsMap[mysteptest]+1;
       //G4StatsMap.erase(G4StatsMap.find(mysteptest));
       G4StatsMap[mysteptest]+=1;
       std::cout << "Incremented already existing StepID "//<<"( c:"<<c<<" "
	 // <<mysteptest.GetPhysicalVolumeName()
	 // <<" "
	 	 <<mysteptest.GetRegionName()
	 	 <<" "
	 	 <<mysteptest.GetProcessName()
	 	 <<" "
	 	 <<mysteptest.GetParticlePDGID()//;
		 <<" Number of Steps: "
		 <<G4StatsMap[mysteptest];
       //		 <<" "
       // <<mysteptest.GetParticleName();
     }
   //Keep the last step to check the last two steps are not the same... (issue with map.end()?)
   //laststep=mysteptest;
   //If the verbose flag is set, then dump the information
   if(m_verbose) {
 
     if (iStep->GetPostStepPoint()->GetPhysicalVolume()) 
       {
	 //std::cout << " MyStepID PhysicalVolume: "<< mysteptest.GetPhysicalVolumeName();
	 std::cout << " MyStepID RegionName: "<< mysteptest.GetRegionName();
       }
     if (iStep->GetTrack()->GetCreatorProcess()) 
       {
	 std::cout << " MyStepID ProcessName: "<< mysteptest.GetProcessName();
       }
     std::cout << " MyStepID ParticlePDGID: "<< mysteptest.GetParticlePDGID();
     //std::cout << " MyStepID ParticleName: "<< mysteptest.GetParticleName();
   }
   std::cout <<std::endl;
 
}
//UPDATE(G4Step)
//UPDATE(EndOfRun)
  void update(const EndOfRun* iRun) {
  std::cout <<"++ signal EndOfRun " <<std::endl;
  //Test maps
  //std::map<strin
  }
  //UPDATE(EndOfEvent)
  void update(const EndOfEvent* iRun) {
  std::cout <<"++ signal EndOfEvent " <<std::endl;
  std::cout <<" G4StatsMap size is: "<<G4StatsMap.size()<<std::endl;
  //std::map<const MyStepID,int>::const_iterator fake;
  int i=0;
  for (std::map<const MyStepID,int>::const_iterator step = G4StatsMap.begin(); step != G4StatsMap.end(); ++step){
    std::cout <<i<<" G4StatsMap step is: "<<step->first.GetRegionName()<<" "<<step->first.GetProcessName()<<" "<<step->first.GetParticlePDGID();
    std::cout <<" Number of such steps: "<< step->second <<std::endl;
    i++;
    //fake=step;
    //fake++;
    //if (step->first == fake->first){
    //  std::cout<<"COMPARISON!"<<std::endl;
    //} 
    //else {
    // std::cout<<"COMPARISON SAYS NOT EQUAL!"<<std::endl;
    //}
  }
}
UPDATE(EndOfTrack)

  private:

//SimTracer(const SimTracer&); // stop default
 
//const SimTracer& operator=(const SimTracer&); // stop default
 
// ---------- member data --------------------------------
 
 bool m_verbose;
//Adding the G4StatsMap to keep track of statistics in terms of step information... 
 std::map<const MyStepID,int> G4StatsMap;
 //MyStepID laststep;
 
};


#endif
