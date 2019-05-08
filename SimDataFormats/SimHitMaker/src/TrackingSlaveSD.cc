#include "SimDataFormats/SimHitMaker/interface/TrackingSlaveSD.h"
//#include "SimG4Core/Application/interface/SimTrackManager.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>
//#define DEBUG

using std::cout;
using std::endl;

TrackingSlaveSD::TrackingSlaveSD(std::string myName) : name_(myName) {
  LogDebug("HitBuildInfo") << " TrackingSlaveSD " << name_;
}

TrackingSlaveSD::~TrackingSlaveSD() {}

void TrackingSlaveSD::Initialize() {
  LogDebug("HitBuildInfo") << " initialize TrackingSlaveSD " << name_;

  hits_.clear();
}
/*
void TrackingSlaveSD::renumbering(const SimTrackManager* tkManager){
  //
  // Now renumber the Hits
  //
  edm::LogInfo("TrackRenumberingInfo")<< " TrackingSlaveSD "<<name()<<"
renumbering " << hits_.size() <<" hits.";
  //
  // now I loop over PSimHits and change the id inside
  //
  for(TrackingSlaveSD::Collection::const_iterator it = begin(); it!= end();
it++){ PSimHit& temp = const_cast<PSimHit&>(*it); unsigned int nt =
tkManager->g4ToSim(temp.trackId());

    LogDebug("TrackRenumberingInfo")<<" Studying PSimHit " << temp
                                    <<" Changing TrackID from " <<
temp.trackId()
                                    <<" with " << nt;

    setTrackId( temp, nt);
  }

}
*/
bool TrackingSlaveSD::format() {
  LogDebug("HitBuildInfo") << " TrackingSlaveSD " << name_ << " formatting " << hits_.size() << " hits.";

  return true;
}

bool TrackingSlaveSD::processHits(const PSimHit &ps) {
  LogDebug("HitBuildInfo") << " Sent Hit " << ps << " to ROU " << name_;

  hits_.push_back(ps);
  return true;
}

void TrackingSlaveSD::setTrackId(PSimHit &hit, unsigned int k) { hit.theTrackId = k; }
