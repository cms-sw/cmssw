#include "SimG4CMS/Tracker/interface/TrackingSlaveSDWithRenumbering.h"
#include "SimG4Core/Application/interface/SimTrackManager.h"

#define DEBUG

#include <iostream>

TrackingSlaveSDWithRenumbering::TrackingSlaveSDWithRenumbering(std::string myName,const SimTrackManager* manager) : 
TrackingSlaveSD(myName),m_trackManager(manager){
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
    unsigned int nt = m_trackManager->g4ToSim(temp.trackId());
#ifdef DEBUG
    std::cout <<" Studying PSimHit " << temp << std::endl;
    std::cout <<" Changing TrackID from " << temp.trackId();
    std::cout <<" with " << nt << std::endl;
#endif
    setTrackId( temp, nt);
  }
}

bool TrackingSlaveSDWithRenumbering::format(){
  
  std::cout << " TrackingSlaveSDWithRenumbering "<<name()<<" formatting " << hits_.size() <<" hits."<< std::endl;
  return true;
}
