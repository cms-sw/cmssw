#include "RecoTracker/TkDetLayers/interface/BoundDiskSector.h"
#include "RecoTracker/TkDetLayers/interface/DiskSectorBounds.h"


float BoundDiskSector::innerRadius() const {
  return dynamic_cast<const DiskSectorBounds&>(bounds()).innerRadius();
}

float BoundDiskSector::outerRadius() const {
  return dynamic_cast<const DiskSectorBounds&>(bounds()).outerRadius();
}

float BoundDiskSector::phiExtension() const {
  return dynamic_cast<const DiskSectorBounds&>(bounds()).phiExtension();
}
