#include "SimDataFormats/SimHitMaker/interface/TrackingSlaveSD.h"

#include <iostream>
#define debug

using std::cout;
using std::endl;

TrackingSlaveSD::TrackingSlaveSD(std::string myName) : name_(myName){
#ifdef debug
 cout << " TrackingSlaveSD " << name_ << endl; 
#endif
}

TrackingSlaveSD::~TrackingSlaveSD()
{}

void TrackingSlaveSD::Initialize(){

#ifdef debug
  std::cout << " initialize TrackingSlaveSD "<< name_ << std::endl;
#endif  
  hits_.clear(); 
}

bool TrackingSlaveSD::format()
{
#ifdef debug 
  cout << " TrackingSlaveSD " << name_<< " formatting " << hits_.size() << " hits." << endl;
#endif  
  return true;
}

bool TrackingSlaveSD::processHits(const PSimHit & ps)
{
#ifdef debug
  std::cout <<" Sent Hit " << ps << " to ROU " << name_ << std::endl;
#endif
  hits_.push_back(ps);
  return true;
} 

void TrackingSlaveSD::setTrackId(PSimHit & hit, unsigned int k)
{ hit.theTrackId = k; }


