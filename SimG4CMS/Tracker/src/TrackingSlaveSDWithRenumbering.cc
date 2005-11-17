#include "SimG4CMS/Tracker/interface/TrackingSlaveSDWithRenumbering.h"
#include "SimG4Core/Notification/interface/EndOfEvent.h"
#include "SimG4Core/Application/interface/EventAction.h"
#include "SimG4CMS/Tracker/interface/TrackerHitsObject.h"


#define DEBUG

#include <iostream>

TrackingSlaveSDWithRenumbering::TrackingSlaveSDWithRenumbering(std::string myName) : TrackingSlaveSD(myName), eventAction(0){
}

void TrackingSlaveSDWithRenumbering::lazyUpDate(const  EventAction * ev){
  eventAction = ev;
}

void TrackingSlaveSDWithRenumbering::update(const EndOfEvent*  ev){
  //
  // Now renumber the Hits
  //
  std::cout << " TrackingSlaveSDWithRenumbering "<<name()<<" renumbering " << hits_.size() <<" hits."<< std::endl;
  //
  // now I loop over PSimHits and change the id inside
  //
  for(TrackingSlaveSD::Collection::const_iterator it = begin(); it!= end(); it++){
    PSimHit& temp = const_cast<PSimHit&>(*it);
    unsigned int nt = eventAction->g4ToSim(temp.trackId());
#ifdef DEBUG
    std::cout <<" Studying PSimHit " << temp << std::endl;
    std::cout <<" Changing TrackID from " << temp.trackId();
    std::cout <<" with " << nt << std::endl;
#endif
    setTrackId( temp, nt);
  }
  //
  // Here I dispatch it
  //
  //  TrackerHitsObject t(name(),hits_);
}

bool TrackingSlaveSDWithRenumbering::format(){
  
  std::cout << " TrackingSlaveSDWithRenumbering "<<name()<<" formatting " << hits_.size() <<" hits."<< std::endl;
  return true;
}
