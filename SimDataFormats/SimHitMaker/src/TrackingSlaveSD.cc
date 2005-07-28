#include "SimHitMakers/TrackingHitsWriter/interface/TrackingSlaveSD.h"

using std::cout;
using std::endl;

TrackingSlaveSD::TrackingSlaveSD(std::string myName) : name_(myName)
{ cout << " TrackingSlaveSD " << name_ << endl; }

TrackingSlaveSD::~TrackingSlaveSD() 
{}

void TrackingSlaveSD::Initialize()
{ hits_.clear(); }

bool TrackingSlaveSD::format()
{
    cout << " TrackingSlaveSD " << name_ 
	 << " formatting " << hits_.size() << " hits." << endl;
    return true;
}

bool TrackingSlaveSD::processHits(const PSimHit & ps)
{
    hits_.push_back(ps);
    return true;
} 

void TrackingSlaveSD::setTrackId(PSimHit & hit, unsigned int k)
{ hit.theTrackId = k; }


