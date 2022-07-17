#include "Phase2EndcapLayerDoubleDisk.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/GeometrySurface/interface/SimpleDiskBounds.h"

#include "TrackingTools/DetLayers/interface/DetLayerException.h"
#include "TrackingTools/GeomPropagators/interface/HelixForwardPlaneCrossing.h"

#include <array>
#include "DetGroupMerger.h"

using namespace std;

const std::vector<const GeometricSearchDet*>& Phase2EndcapLayerDoubleDisk::components() const {
  throw cms::Exception("Phase2EndcapLayerDoubleDisk::components() is not implemented");
}

void Phase2EndcapLayerDoubleDisk::fillSubDiskPars(int i) {
  const BoundDisk& subDiskDisk = static_cast<const BoundDisk&>(theComps[i]->surface());
  SubDiskPar tempPar;
  tempPar.theSubDiskZ = std::abs(subDiskDisk.position().z());
  subDiskPars.push_back(tempPar);
}

Phase2EndcapLayerDoubleDisk::Phase2EndcapLayerDoubleDisk(vector<const Phase2EndcapSubDisk*>& subDisks)
    : RingedForwardLayer(true), theComponents{nullptr} {
  theSubDisksSize = subDisks.size();
  LogDebug("TkDetLayers") << "Number of subdisks in Phase2 IT EC layer is " << theSubDisksSize << std::endl;
  setSurface(computeDisk(subDisks));

  for (unsigned int i = 0; i != subDisks.size(); ++i) {
    theComps.push_back(subDisks[i]);
    fillSubDiskPars(i);
    theBasicComps.insert(
        theBasicComps.end(), (*subDisks[i]).basicComponents().begin(), (*subDisks[i]).basicComponents().end());
  }

  LogDebug("TkDetLayers") << "==== DEBUG Phase2EndcapLayer =====";
  LogDebug("TkDetLayers") << "r,zed pos  , thickness, innerR, outerR: " << this->position().perp() << " , "
                          << this->position().z() << " , " << this->specificSurface().bounds().thickness() << " , "
                          << this->specificSurface().innerRadius() << " , " << this->specificSurface().outerRadius();
}

BoundDisk* Phase2EndcapLayerDoubleDisk::computeDisk(const vector<const Phase2EndcapSubDisk*>& subDisks) const {
  return tkDetUtil::computeDisk(subDisks);
}

Phase2EndcapLayerDoubleDisk::~Phase2EndcapLayerDoubleDisk() {
  for (auto c : theComps)
    delete c;

  delete theComponents.load();
}

void Phase2EndcapLayerDoubleDisk::groupedCompatibleDetsV(const TrajectoryStateOnSurface& startingState,
                                                         const Propagator& prop,
                                                         const MeasurementEstimator& est,
                                                         std::vector<DetGroup>& result) const {
  std::array<int, 2> const& subDiskIndices = subDiskIndicesByCrossingProximity(startingState, prop);

  //order subdisks in z
  //Subdisk near in z: 0, Subdisk far in z: 1
  std::vector<int> subDiskOrder(theSubDisksSize);
  std::fill(subDiskOrder.begin(), subDiskOrder.end(), 1);
  if (theSubDisksSize > 1) {
    if (std::abs(theComps[0]->position().z()) < std::abs(theComps[1]->position().z())) {
      for (int i = 0; i < theSubDisksSize; i++) {
        if (i % 2 == 0)
          subDiskOrder[i] = 0;
      }
    } else if (std::abs(theComps[0]->position().z()) > std::abs(theComps[1]->position().z())) {
      std::fill(subDiskOrder.begin(), subDiskOrder.end(), 0);
      for (int i = 0; i < theSubDisksSize; i++) {
        if (i % 2 == 0)
          subDiskOrder[i] = 1;
      }
    } else {
      throw DetLayerException("SubDisks in Endcap Layer have same z position, no idea how to order them!");
    }
  }

  auto index = [&subDiskIndices, &subDiskOrder](int i) { return subDiskOrder[subDiskIndices[i]]; };

  std::vector<DetGroup> closestResult;
  theComps[subDiskIndices[0]]->groupedCompatibleDetsV(startingState, prop, est, closestResult);
  // if the closest is empty, use the next one and exit: inherited from TID !
  if (closestResult.empty()) {
    theComps[subDiskIndices[1]]->groupedCompatibleDetsV(startingState, prop, est, result);
    return;
  }

  // check if next subdisk is found

  bool subdisk1ok = subDiskIndices[1] != -1;

  // determine if we are propagating from in to out (0) or from out to in (1)

  int direction = 0;
  if (startingState.globalPosition().z() * startingState.globalMomentum().z() > 0) {
    if (prop.propagationDirection() == alongMomentum)
      direction = 0;
    else
      direction = 1;
  } else {
    if (prop.propagationDirection() == alongMomentum)
      direction = 1;
    else
      direction = 0;
  }

  if (index(0) == index(1)) {
    if (subdisk1ok) {
      std::vector<DetGroup> subdisk1res;
      theComps[subDiskIndices[1]]->groupedCompatibleDetsV(startingState, prop, est, subdisk1res);
      DetGroupMerger::addSameLevel(std::move(subdisk1res), closestResult);
      result.swap(closestResult);
      return;
    }
  } else {
    std::vector<DetGroup> subdisk1res;
    if (subdisk1ok) {
      theComps[subDiskIndices[1]]->groupedCompatibleDetsV(startingState, prop, est, subdisk1res);
    }
    if (!subdisk1res.empty()) {
      DetGroupMerger::orderAndMergeTwoLevels(
          std::move(closestResult), std::move(subdisk1res), result, index(0), direction);
      return;
    } else {
      result.swap(closestResult);
      return;
    }
  }
}

std::array<int, 2> Phase2EndcapLayerDoubleDisk::subDiskIndicesByCrossingProximity(
    const TrajectoryStateOnSurface& startingState, const Propagator& prop) const {
  typedef HelixForwardPlaneCrossing Crossing;
  typedef MeasurementEstimator::Local2DVector Local2DVector;

  HelixPlaneCrossing::PositionType startPos(startingState.globalPosition());
  HelixPlaneCrossing::DirectionType startDir(startingState.globalMomentum());
  PropagationDirection propDir(prop.propagationDirection());
  float rho(startingState.transverseCurvature());

  // calculate the crossings with the subdisk surfaces

  Crossing myXing(startPos, startDir, rho, propDir);

  std::vector<GlobalPoint> subDiskCrossings;
  subDiskCrossings.reserve(theSubDisksSize);

  for (int i = 0; i < theSubDisksSize; i++) {
    const BoundDisk& theSubDisk = static_cast<const BoundDisk&>(theComps[i]->surface());
    pair<bool, double> pathlen = myXing.pathLength(theSubDisk);
    if (pathlen.first) {
      subDiskCrossings.push_back(GlobalPoint(myXing.position(pathlen.second)));
    } else {
      // TO FIX.... perhaps there is something smarter to do
      subDiskCrossings.push_back(GlobalPoint(0., 0., 0.));
    }
  }

  //find two closest subdisks to the crossing

  return findTwoClosest(subDiskCrossings);
}

std::array<int, 2> Phase2EndcapLayerDoubleDisk::findTwoClosest(std::vector<GlobalPoint> subDiskCrossing) const {
  std::array<int, 2> theBins = {{-1, -1}};
  theBins[0] = 0;
  float initialZ = subDiskPars[0].theSubDiskZ;
  float zDiff0 = std::abs(subDiskCrossing[0].z() - initialZ);
  float zDiff1 = -1.;
  for (int i = 1; i < theSubDisksSize; i++) {
    float subDiskZ = subDiskPars[i].theSubDiskZ;
    float testDiff = std::abs(subDiskCrossing[i].z() - subDiskZ);
    if (testDiff < zDiff0) {
      zDiff1 = zDiff0;
      zDiff0 = testDiff;
      theBins[1] = theBins[0];
      theBins[0] = i;
    } else if (zDiff1 < 0 || testDiff < zDiff1) {
      zDiff1 = testDiff;
      theBins[1] = i;
    }
  }

  return theBins;
}
