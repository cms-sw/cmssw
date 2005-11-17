#include "SimG4CMS/Tracker/interface/TkSimTrackSelection.h"
#include "SimG4Core/Notification/interface/TrackInformation.h"
#include <iostream>
#include "G4Track.hh"

#define DEBUG
#define DUMPPROCESSES

#ifdef DUMPPROCESSES
#include "G4VProcess.hh"
#endif

TkSimTrackSelection::TkSimTrackSelection( edm::ParameterSet const & p) : rTracker(1200.*mm), zTracker(3000.*mm) {
  edm::ParameterSet m_SimTrack = p.getParameter<edm::ParameterSet>("SimTrack");
  energyCut = m_SimTrack.getParameter<double>("EnergyThresholdForPersistencyInGeV")*GeV; //default must be 0.5
  energyHistoryCut  = m_SimTrack.getParameter<bool>("EnergyThresholdForHistoryInGeV")*GeV;//default must be 00.5

  std::cout <<"Criteria for Saving Tracker SimTracks:  ";
  std::cout <<" History: "<<energyHistoryCut<< " MeV ; Persistency: "<< energyCut<<" MeV "<<std::endl;
}

void TkSimTrackSelection::update(const BeginOfTrack *bot){
  const G4Track* gTrack = (*bot)();
#ifdef DUMPPROCESSES
  std::cout <<" -> process creator pointer "<<gTrack->GetCreatorProcess()<<std::endl;
  if (gTrack->GetCreatorProcess())
    std::cout <<" -> PROCESS CREATOR : "<<gTrack->GetCreatorProcess()->GetProcessName()<<std::endl;

#endif


  //
  //Position
  //
  const G4ThreeVector pos = gTrack->GetPosition();
#ifdef DEBUG
  std::cout <<" ENERGY MeV "<<gTrack->GetKineticEnergy()<<" Energy Cut" << energyCut<<std::endl;
  std::cout <<" TOTAL ENERGY "<<gTrack->GetTotalEnergy()<<std::endl;
  std::cout <<" WEIGHT "<<gTrack->GetWeight()<<std::endl;
#endif
  //
  // Check if in Tracker Volume
  //
  if (pos.perp() < rTracker && abs(pos.z()) < zTracker){
    //
    // inside the Tracker
    //
#ifdef DEBUG
      std::cout <<" INSIDE TRACKER"<<std::endl;
#endif
    if (gTrack->GetKineticEnergy() > energyCut){
      TrackInformation* info = getOrCreateTrackInformation(gTrack);
#ifdef DEBUG
      std::cout <<" POINTER "<<info<<std::endl;
      std::cout <<" track inside the tracker selected for STORE"<<std::endl;
#endif
      info->storeTrack(true);
    }
    //
    // Save History?
    //
    if (gTrack->GetKineticEnergy() > energyHistoryCut){
      TrackInformation* info = getOrCreateTrackInformation(gTrack);
      info->putInHistory();
#ifdef DEBUG
      std::cout <<" POINTER "<<info<<std::endl;
      std::cout <<" track inside the tracker selected for HISTORY"<<std::endl;
#endif
    }
    
  }
}

TrackInformation* TkSimTrackSelection::getOrCreateTrackInformation( const G4Track* gTrack){
  G4VUserTrackInformation* temp = gTrack->GetUserInformation();
  if (temp == 0){
    std::cout <<" ERROR: no G4VUserTrackInformation available"<<std::endl;
    abort();
  }else{
    TrackInformation* info = dynamic_cast<TrackInformation*>(temp);
    if (info ==0){
      std::cout <<" ERROR: TkSimTrackSelection: the UserInformation does not appear to be a TrackInformation"<<std::endl;
      abort();
    }
    return info;
  }
}

