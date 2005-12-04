#include "SimG4CMS/Muon/interface/MuonSlaveSD.h"
#include "Geometry/MuonBaseAlgo/interface/MuonSubDetector.h"

#include "SimG4Core/Notification/interface/EndOfEvent.h"
#include "SimG4Core/Application/interface/SimTrackManager.h"
#include "Geometry/MuonBaseAlgo/interface/MuBarIDPacking.h"

#include <iostream>

//#define DEBUG


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
#ifdef DEBUG
    std::cout << " MuonSlaveSD::clearHits "<< detector->Name() << std::endl;
#endif
    hits_.clear();
}

bool MuonSlaveSD::format()
{
#ifdef DEBUG
  std::cout << " MuonSlaveSD "<<detector->Name()<<" formatting " << hits_.size() <<" hits."<< std::endl;
#endif
  if (detector->isBarrel()) {
    sort(hits_.begin(),hits_.end(), FormatBarrelHits());
  } else if (detector->isEndcap()) {
    sort(hits_.begin(),hits_.end(), FormatEndcapHits());
  } else if (detector->isRpc()) {
    sort(hits_.begin(),hits_.end(), FormatRpcHits());
  } 
  
  return true;
}


void MuonSlaveSD::update(const  EndOfEvent * ev)
{
  //
  // Now renumber the Hits
  //
#ifdef DEBUG
  std::cout << " MuonSlaveSD renumbering " << name() << " " << hits_.size() <<" hits."<< std::endl;
#endif
 //?? check();
  //
  // now I loop over PSimHits and change the id inside
  //
  for(MuonSlaveSD::const_iterator it = begin(); it!=end(); it++){
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

bool FormatBarrelHits::operator() (const PSimHit & a, const PSimHit & b)
{
  return (sortId(a)<sortId(b));
}

int FormatBarrelHits::sortId(const PSimHit & a)  const 
{
  MuBarIDPacking packing;
  int duId = a.detUnitId();
  return packing.sector(duId)
         +100*packing.stat(duId)
         +1000*packing.wheel(duId)
         +10000*packing.slayer(duId)
         +100000*packing.layer(duId);
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

