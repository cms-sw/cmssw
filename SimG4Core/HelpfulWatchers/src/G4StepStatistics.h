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
// $Id: G4StepStatistics.h,v 1.2 2009/02/25 16:14:14 gbenelli Exp $
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
#include <FWCore/ServiceRegistry/interface/Service.h>
#include <PhysicsTools/UtilAlgos/interface/TFileService.h>

#include <TROOT.h>
#include <TTree.h>
#include <TFile.h>
#include <TVector.h>
#include <TString.h>
#include <TClonesArray.h>
//#include<TObjString.h>

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

//Define a class StepID
class StepID {

 private:
  //G4 Region
  G4String theG4RegionName;
  //G4 Physical Process
  G4String theG4ProcessName;
  //Particle PDG ID
  G4int theParticlePDGID;
  //Track Number
  //G4int theTrackID;
  
 public:
  //Constructor using G4Step
  StepID(const G4Step* theG4Step)
    :
    theG4RegionName("UNDEFINED"),
    theG4ProcessName("UNDEFINED"),
    theParticlePDGID(theG4Step->GetTrack()->GetDefinition()->GetPDGEncoding())//,
    //theTrackID(theG4Step->GetTrack()->GetTrackID())
    {
      if (theG4Step->GetPostStepPoint()->GetPhysicalVolume()) {
	  theG4RegionName = theG4Step->GetPostStepPoint()->GetPhysicalVolume()->GetLogicalVolume()->GetRegion()->GetName();
	}
      if (theG4Step->GetTrack()->GetCreatorProcess()){
	  theG4ProcessName = theG4Step->GetTrack()->GetCreatorProcess()->GetProcessName();
	}
      
    }

  //Getters
  //G4String GetPhysicalVolumeName () const { return theG4PhysicalVolume; }
  G4String GetRegionName () const { return theG4RegionName; }
  G4String GetProcessName () const { return theG4ProcessName; }
  G4int GetParticlePDGID () const { return theParticlePDGID; }
  //G4int GetTrackID () const { return theTrackID; }

  //Comparison Operators (necessary in order to use StepID as a key in a map)                                                                                                                  
  bool operator==(const StepID& id) const{
    //std::cout<<"Operator == (StepID)"<<std::endl;
    //std::cout<<" Present = "<<theG4RegionName<<" New = "<<id.GetRegionName()<<" Comparator = "<<strcmp(theG4RegionName,id.GetRegionName())<<
    // " Present = "<<theG4ProcessName<<" New = "<<id.GetProcessName()<<" Comparator = "<<strcmp(theG4ProcessName,id.GetProcessName())<<
    // " Present = "<<theParticlePDGID<<" New = "<<id.GetParticlePDGID()<<std::endl;

    return ( strcmp(theG4RegionName,id.GetRegionName())==0 && strcmp(theG4ProcessName,id.GetProcessName())==0 && theParticlePDGID==id.GetParticlePDGID() ) ? true : false;//&& theTrackID==id.GetTrackID() ) ? true : false;
  }

  bool operator<(const StepID& id) const
    {
      //std::cout<<"Operator < (StepID)"<<std::endl;
      //std::cout<<" Present = "<<theG4RegionName<<" New = "<<id.GetRegionName()<<" Comparator = "<<strcmp(theG4RegionName,id.GetRegionName())<<
      // " Present = "<<theG4ProcessName<<" New = "<<id.GetProcessName()<<" Comparator = "<<strcmp(theG4ProcessName,id.GetProcessName())<<
      // " Present = "<<theParticlePDGID<<" New = "<<id.GetParticlePDGID()<<std::endl;

      if (theParticlePDGID != id.GetParticlePDGID()){
	//std::cout<<"Return "<<(theParticlePDGID > id.GetParticlePDGID());
	return (theParticlePDGID > id.GetParticlePDGID());
      }
      //if (theTrackID != id.GetTrackID()){
	//std::cout<<"Return "<<(theTrackID > id.GetTrackID());
	//return (theTrackID > id.GetTrackID());
      //}
      else if (strcmp(theG4RegionName,id.GetRegionName())!=0){
	//std::cout<<"Return "<<strcmp(theG4RegionName,id.GetRegionName())>0 ? true : false;
	return strcmp(theG4RegionName,id.GetRegionName())>0 ? true : false;
      }
      else if (strcmp(theG4ProcessName,id.GetProcessName())!=0){
	//std::cout<<"Return "<<strcmp(theG4ProcessName,id.GetProcessName())>0 ? true : false;
	return strcmp(theG4ProcessName,id.GetProcessName())>0 ? true : false;
      }
      else {//The case in which they are all equal!
	//std::cout<<"Return "<< false;
	return false;
      }
    }

  bool operator>(const StepID& id) const
    {
      //std::cout<<"Operator > (StepID)"<<std::endl;
      //std::cout<<" Present = "<<theG4RegionName<<" New = "<<id.GetRegionName()<<" Comparator = "<<strcmp(theG4RegionName,id.GetRegionName())<<
      // " Present = "<<theG4ProcessName<<" New = "<<id.GetProcessName()<<" Comparator = "<<strcmp(theG4ProcessName,id.GetProcessName())<<
      // " Present = "<<theParticlePDGID<<" New = "<<id.GetParticlePDGID()<<std::endl;

      if(theParticlePDGID != id.GetParticlePDGID()){
        return (theParticlePDGID < id.GetParticlePDGID());
      }
      //if (theTrackID != id.GetTrackID()){
	//std::cout<<"Return "<<(theTrackID < id.GetTrackID());
	//return (theTrackID < id.GetTrackID());
      //}
      else if(strcmp(theG4RegionName,id.GetRegionName())!=0){
        return strcmp(theG4RegionName,id.GetRegionName())<0 ? true : false;
      }
      else if (strcmp(theG4ProcessName,id.GetProcessName())!=0){
        return strcmp(theG4ProcessName,id.GetProcessName())<0 ? true : false;
      }
      else {//The case in which they are all equal!
	return false;
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
     m_verbose(pSet.getUntrackedParameter<bool>("verbose",false)),
     Event(0)
     {
       //Adding TFile Service output
       std::cout<<"Start of the constuctor!!!!!!!!"<<std::endl;
       G4StepTree = fs->make<TTree>("G4StepTree","G4Step Tree ");
       G4StepTree->Branch("Event",&Event,"Event/I");
       //std::cout<<"created Branch for Event!"<<std::endl;
       //TrackID = new TArrayI(100000);
       //std::cout<<"Allocated TrackID!"<<std::endl;
       //G4StepTree->Branch("TrackID",&TrackID,100000);
       //std::cout<<"Allocated TrackID and created Branch for it!"<<std::endl;
       PDGID = new TArrayI(100000);
       //std::cout<<"Allocated PDGID!"<<std::endl;
       G4StepTree->Branch("PDGID",&PDGID,100000);
       //std::cout<<"Allocated PDGID and created Branch for it!"<<std::endl;
       Region = new TClonesArray("TObjString",100000);
       G4StepTree->Branch("Region",&Region,100000);
       //std::cout<<"Allocated Region and created Branch for it!"<<std::endl;
       Process = new TClonesArray("TObjString",100000);
       G4StepTree->Branch("Process",&Process,100000);
       //std::cout<<"Allocated Process and created Branch for it!"<<std::endl;
       G4StepFreq = new TArrayI(100000);
       G4StepTree->Branch("G4StepFreq",&G4StepFreq,100000);
       //std::cout<<"Allocated G4StepFreq and created Branch for it!"<<std::endl;
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
   StepID mysteptest(iStep);
   //Add the StepID to the map, or increment if it already exists:
   if ( G4StatsMap.find(mysteptest) == G4StatsMap.end() )
     {
       // std::cout<<"ADDED STEP!"<<std::endl;
       //Allocating new memory for a pointer to associate with the key mysteptest
       //in our map. Initializing it to 1,will be incremented working on the value of the pointer.
       unsigned int* MyValue = new unsigned int(1);
       //Inserting the new key,value pair
       G4StatsMap.insert(std::make_pair(mysteptest, MyValue));
       //std::cout << "Added to map StepID "
     }
   else
     {
       //Incrementing the value of the pointer by 1
       *G4StatsMap[mysteptest] = *G4StatsMap[mysteptest] + 1;
       //std::cout << "Incremented already existing StepID "
        }
   
   //If the verbose flag is set, then dump the information
   if (m_verbose) {
     if (iStep->GetPostStepPoint()->GetPhysicalVolume()) {
       std::cout << " StepID RegionName: "<< mysteptest.GetRegionName();
     }
     if (iStep->GetTrack()->GetCreatorProcess()) {
       std::cout << " StepID ProcessName: "<< mysteptest.GetProcessName();
     }
     std::cout << " StepID ParticlePDGID: "<< mysteptest.GetParticlePDGID();
   }
   std::cout <<std::endl;
}
//UPDATE(G4Step)
UPDATE(EndOfRun)
  //  void update(const EndOfRun* iRun) {
  //std::cout <<"++ signal EndOfRun " <<std::endl;
  //}
  
  //UPDATE(EndOfEvent)
  void update(const EndOfEvent* iRun) {
  std::cout <<"++ signal EndOfEvent " <<std::endl;
  Event++;
  
  //Dumping the map in the log if verbose is chosen:
  if(m_verbose) {
    std::cout <<" G4StatsMap size is: "<<G4StatsMap.size()<<std::endl;
  }
  int index(0);
  for (std::map<const StepID,unsigned int*>::const_iterator step = G4StatsMap.begin(); step != G4StatsMap.end(); ++step, ++index){
    if(m_verbose) {
      std::cout <<" G4StatsMap step is: "<<step->first.GetRegionName()<<" "<<step->first.GetProcessName()<<" "<<step->first.GetParticlePDGID();//<<" "<<step->first.GetTrackID() ;
      std::cout <<" Number of such steps: "<< *step->second <<std::endl;
    }
    //Rolling the map into 5 "arrays", containing the StepID information and the G4Step statistics
    //(*TrackID)[index]=step->first.GetTrackID();
    //std::cout<<"Assigned TrackID"<<std::endl;
    (*PDGID)[index]=step->first.GetParticlePDGID();
    //std::cout<<"Assigned PDGID"<<std::endl;
    new ((*Region)[index]) TObjString (step->first.GetRegionName());
    //std::cout<<"Assigned Region"<<std::endl;
    new ((*Process)[index]) TObjString (step->first.GetProcessName());
    //std::cout<<"Assigned Process"<<std::endl;
    (*G4StepFreq)[index]=*step->second;
    //std::cout<<"Assigned G4StepFreq"<<std::endl;
  }
  
  G4StepTree->Fill();
}
UPDATE(EndOfTrack)

  private:
 
 bool m_verbose;
 
//Adding the G4StatsMap to keep track of statistics in terms of step information... 
 std::map<const StepID,unsigned int*> G4StatsMap;
 edm::Service<TFileService> fs;
 TTree* G4StepTree;
 unsigned int Event;
 //TArrayI* TrackID;
 TArrayI* PDGID;
 TClonesArray* Region;
 TClonesArray* Process;
 TArrayI* G4StepFreq;
};

#endif
