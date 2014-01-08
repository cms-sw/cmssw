#include "SimG4CMS/Muon/interface/MuonSlaveSD.h"
#include "Geometry/MuonNumbering/interface/MuonSubDetector.h"

#include "SimG4Core/Application/interface/SimTrackManager.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>


MuonSlaveSD::MuonSlaveSD(MuonSubDetector* d,
			 const SimTrackManager* manager): 
  TrackingSlaveSD(d->name() ), m_trackManager(manager)
{
  detector=d;

}

MuonSlaveSD::~MuonSlaveSD() { 
}

void MuonSlaveSD::clearHits()
{
  LogDebug("MuonSimDebug") << " MuonSlaveSD::clearHits "<< detector->name() << std::endl;
  hits_.clear();
}

bool MuonSlaveSD::format()
{
  LogDebug("MuonSimDebug") << " MuonSlaveSD "<<detector->name()<<" formatting " << hits_.size() <<" hits."<< std::endl;
  if (detector->isBarrel()) {
    sort(hits_.begin(),hits_.end(), FormatBarrelHits());
  } else if (detector->isEndcap()) {
    sort(hits_.begin(),hits_.end(), FormatEndcapHits());
  } else if (detector->isRpc()) {
    sort(hits_.begin(),hits_.end(), FormatRpcHits());
  } else if (detector->isGem()) {
    sort(hits_.begin(),hits_.end(), FormatGemHits());
  } else if (detector->isME0()) {
    sort(hits_.begin(),hits_.end(), FormatMe0Hits());
  } 
  return true;
}


bool FormatBarrelHits::operator() (const PSimHit & a, const PSimHit & b)
{
  return (sortId(a)<sortId(b));
}

int FormatBarrelHits::sortId(const PSimHit & a)  const 
{
  return a.detUnitId();
}

bool FormatEndcapHits::operator() (const PSimHit & a, const PSimHit & b)
{
  return (sortId(a)<sortId(b));
}

int FormatEndcapHits::sortId(const PSimHit & a)  const 
{
  return a.detUnitId();
}

bool FormatRpcHits::operator() (const PSimHit & a, const PSimHit & b)
{
  return (sortId(a)<sortId(b));
}

int FormatRpcHits::sortId(const PSimHit & a)  const 
{
  return a.detUnitId();
}

bool FormatGemHits::operator() (const PSimHit & a, const PSimHit & b)
{
  return (sortId(a)<sortId(b));
}

int FormatGemHits::sortId(const PSimHit & a)  const 
{
  return a.detUnitId();
}


bool FormatMe0Hits::operator() (const PSimHit & a, const PSimHit & b)
{
  return (sortId(a)<sortId(b));
}

int FormatMe0Hits::sortId(const PSimHit & a)  const 
{
  return a.detUnitId();
}
