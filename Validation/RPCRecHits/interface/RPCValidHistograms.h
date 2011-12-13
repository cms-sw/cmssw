#ifndef Validation_RPCRecHits_RPCValidHistograms_H
#define Validation_RPCRecHits_RPCValidHistograms_H

#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include <string>

struct RPCValidHistograms
{
  typedef MonitorElement* MEP;

  RPCValidHistograms()
  {
    booked_ = false;
  };

  void bookHistograms(DQMStore* dbe, const std::string subDir);

  // Hit properties
  MEP clusterSize, clusterSizeBarrel, clusterSizeEndcap;
  MEP avgClusterSize, avgClusterSizeBarrel, avgClusterSizeEndcap;

  MEP nRefHitBarrel, nRefHitEndcap;
  MEP nRecHitBarrel, nRecHitEndcap;
  MEP nMatchHitBarrel, nMatchHitEndcap;

  // Occupancy 1D
  MEP refHitBarrelOccupancy_wheel, refHitEndcapOccupancy_disk, refHitBarrelOccupancy_station;
  MEP recHitBarrelOccupancy_wheel, recHitEndcapOccupancy_disk, recHitBarrelOccupancy_station;
  MEP matchBarrelOccupancy_wheel, matchEndcapOccupancy_disk, matchBarrelOccupancy_station;
  MEP umBarrelOccupancy_wheel, umEndcapOccupancy_disk, umBarrelOccupancy_station;

  // Occupancy 2D
  MEP refHitBarrelOccupancy_wheel_station, refHitEndcapOccupancy_disk_ring;
  MEP recHitBarrelOccupancy_wheel_station, recHitEndcapOccupancy_disk_ring;
  MEP matchBarrelOccupancy_wheel_station, matchEndcapOccupancy_disk_ring;
  MEP umBarrelOccupancy_wheel_station, umEndcapOccupancy_disk_ring;

  // Residuals
  MEP resBarrel, resEndcap;
  MEP res_wheel_res, res_disk_res, res_station_res, res_ring_res;
  
  // Pulls
  MEP pullBarrel, pullEndcap;
  MEP pull_wheel_pull, pull_disk_pull, pull_station_pull, pull_ring_pull;

private:
  bool booked_;
};

#endif

