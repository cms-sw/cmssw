#include "SimDataFormats/SimHitMaker/interface/TrackingSlaveSD.h"
#include "SimG4Core/Application/interface/SimTrackManager.h"

#include <iostream>
//#define DEBUG

using std::cout;
using std::endl;

TrackingSlaveSD::TrackingSlaveSD(std::string myName) : name_(myName){
#ifdef DEBUG
 cout << " TrackingSlaveSD " << name_ << endl; 
#endif
}

TrackingSlaveSD::~TrackingSlaveSD()
{}

void TrackingSlaveSD::Initialize(){

#ifdef DEBUG
  std::cout << " initialize TrackingSlaveSD "<< name_ << std::endl;
#endif  
  hits_.clear(); 
}

void TrackingSlaveSD::renumbering(const SimTrackManager* tkManager){
  //
  // Now renumber the Hits
  //
  std::cout << " TrackingSlaveSD "<<name()<<" renumbering " << hits_.size() <<" hits."<< std::endl;
  //
  // now I loop over PSimHits and change the id inside
  //
  for(TrackingSlaveSD::Collection::const_iterator it = begin(); it!= end(); it++){
    PSimHit& temp = const_cast<PSimHit&>(*it);
    unsigned int nt = tkManager->g4ToSim(temp.trackId());
#ifdef DEBUG
    std::cout <<" Studying PSimHit " << temp << std::endl;
    std::cout <<" Changing TrackID from " << temp.trackId();
    std::cout <<" with " << nt << std::endl;
#endif
    setTrackId( temp, nt);
  }

}
bool TrackingSlaveSD::format()
{
#ifdef DEBUG 
  cout << " TrackingSlaveSD " << name_<< " formatting " << hits_.size() << " hits." << endl;
#endif  
  return true;
}

bool TrackingSlaveSD::processHits(const PSimHit & ps)
{
#ifdef DEBUG
  std::cout <<" Sent Hit " << ps << " to ROU " << name_ << std::endl;
#endif
  hits_.push_back(ps);
  return true;
} 

void TrackingSlaveSD::setTrackId(PSimHit & hit, unsigned int k)
{ hit.theTrackId = k; }


