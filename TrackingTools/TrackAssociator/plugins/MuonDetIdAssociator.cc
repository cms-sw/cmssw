// -*- C++ -*-
//
// Package:    TrackAssociator
// Class:      MuonDetIdAssociator
//
/*

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Dmytro Kovalskyi
//         Created:  Fri Apr 21 10:59:41 PDT 2006
//
//

#include "MuonDetIdAssociator.h"
// #include "Utilities/Timing/interface/TimerStack.h"
#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixPropagator.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "Geometry/DTGeometry/interface/DTChamber.h"
#include "Geometry/CSCGeometry/interface/CSCChamber.h"
#include "Geometry/RPCGeometry/interface/RPCChamber.h"
#include <deque>
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include "Geometry/GEMGeometry/interface/ME0Geometry.h"

void MuonDetIdAssociator::check_setup() const {
  if (geometry_ == nullptr)
    throw cms::Exception("ConfigurationProblem") << "GlobalTrackingGeomtry is not set\n";
  if (cscbadchambers_ == nullptr)
    throw cms::Exception("ConfigurationProblem") << "CSCBadChambers is not set\n";
  DetIdAssociator::check_setup();
}

const GeomDet* MuonDetIdAssociator::getGeomDet(const DetId& id) const {
  if (geometry_ == nullptr)
    throw cms::Exception("ConfigurationProblem") << "GlobalTrackingGeomtry is not set\n";
  const GeomDet* gd = geometry_->idToDet(id);
  if (gd == nullptr)
    throw cms::Exception("NoGeometry") << "Cannot find GeomDet for DetID: " << id.rawId() << "\n";
  return gd;
}

GlobalPoint MuonDetIdAssociator::getPosition(const DetId& id) const {
  Surface::PositionType point(getGeomDet(id)->surface().position());
  return GlobalPoint(point.x(), point.y(), point.z());
}

void MuonDetIdAssociator::getValidDetIds(unsigned int subDectorIndex, std::vector<DetId>& validIds) const {
  validIds.clear();
  if (geometry_ == nullptr)
    throw cms::Exception("ConfigurationProblem") << "GlobalTrackingGeomtry is not set\n";
  if (subDectorIndex != 0)
    throw cms::Exception("FatalError")
        << "Muon sub-dectors are all handle as one sub-system, but subDetectorIndex is not zero.\n";

  // CSC
  if (!geometry_->slaveGeometry(CSCDetId()))
    throw cms::Exception("FatalError") << "Cannnot CSCGeometry\n";
  auto const& geomDetsCSC = geometry_->slaveGeometry(CSCDetId())->dets();
  for (auto it = geomDetsCSC.begin(); it != geomDetsCSC.end(); ++it)
    if (auto csc = dynamic_cast<const CSCChamber*>(*it)) {
      if ((!includeBadChambers_) && (cscbadchambers_->isInBadChamber(CSCDetId(csc->id()))))
        continue;
      validIds.push_back(csc->id());
    }

  // DT
  if (!geometry_->slaveGeometry(DTChamberId()))
    throw cms::Exception("FatalError") << "Cannnot DTGeometry\n";
  auto const& geomDetsDT = geometry_->slaveGeometry(DTChamberId())->dets();
  for (auto it = geomDetsDT.begin(); it != geomDetsDT.end(); ++it)
    if (auto dt = dynamic_cast<const DTChamber*>(*it))
      validIds.push_back(dt->id());

  // RPC
  if (!geometry_->slaveGeometry(RPCDetId()))
    throw cms::Exception("FatalError") << "Cannnot RPCGeometry\n";
  auto const& geomDetsRPC = geometry_->slaveGeometry(RPCDetId())->dets();
  for (auto it = geomDetsRPC.begin(); it != geomDetsRPC.end(); ++it)
    if (auto rpc = dynamic_cast<const RPCChamber*>(*it)) {
      std::vector<const RPCRoll*> rolls = (rpc->rolls());
      for (std::vector<const RPCRoll*>::iterator r = rolls.begin(); r != rolls.end(); ++r)
        validIds.push_back((*r)->id().rawId());
    }

  // GEM
  if (includeGEM_) {
    if (!geometry_->slaveGeometry(GEMDetId()))
      throw cms::Exception("FatalError") << "Cannnot GEMGeometry\n";
    auto const& geomDetsGEM = geometry_->slaveGeometry(GEMDetId())->dets();
    for (auto it = geomDetsGEM.begin(); it != geomDetsGEM.end(); ++it) {
      if (auto gem = dynamic_cast<const GEMSuperChamber*>(*it)) {
        if (gem->id().station() == 0)
          validIds.push_back(gem->id());
        else
          for (auto ch : gem->chambers())
            validIds.push_back(ch->id());
      }
    }
  }
  // ME0
  if (includeME0_) {
    if (!geometry_->slaveGeometry(ME0DetId()))
      throw cms::Exception("FatalError") << "Cannnot ME0Geometry\n";
    auto const& geomDetsME0 = geometry_->slaveGeometry(ME0DetId())->dets();
    for (auto it = geomDetsME0.begin(); it != geomDetsME0.end(); ++it) {
      if (auto me0 = dynamic_cast<const ME0Chamber*>(*it)) {
        validIds.push_back(me0->id());
      }
    }
  }
}

bool MuonDetIdAssociator::insideElement(const GlobalPoint& point, const DetId& id) const {
  if (geometry_ == nullptr)
    throw cms::Exception("ConfigurationProblem") << "GlobalTrackingGeomtry is not set\n";
  LocalPoint lp = geometry_->idToDet(id)->toLocal(point);
  return geometry_->idToDet(id)->surface().bounds().inside(lp);
}

std::pair<DetIdAssociator::const_iterator, DetIdAssociator::const_iterator> MuonDetIdAssociator::getDetIdPoints(
    const DetId& id, std::vector<GlobalPoint>& points) const {
  points.clear();
  points.reserve(8);
  const GeomDet* geomDet = getGeomDet(id);

  // the coners of muon detector elements are not stored and can be only calculated
  // based on methods defined in the interface class Bounds:
  //   width() - x
  //   length() - y
  //   thinkness() - z
  // NOTE: this convention is implementation specific and can fail. Both
  //       RectangularPlaneBounds and TrapezoidalPlaneBounds use it.
  // Even though the CSC geomtry is more complicated (trapezoid),  it's enough
  // to estimate which bins should contain this element. For the distance
  // calculation from the edge, we will use exact geometry to get it right.

  const Bounds* bounds = &(geometry_->idToDet(id)->surface().bounds());
  points.push_back(
      geomDet->toGlobal(LocalPoint(+bounds->width() / 2, +bounds->length() / 2, +bounds->thickness() / 2)));
  points.push_back(
      geomDet->toGlobal(LocalPoint(-bounds->width() / 2, +bounds->length() / 2, +bounds->thickness() / 2)));
  points.push_back(
      geomDet->toGlobal(LocalPoint(+bounds->width() / 2, -bounds->length() / 2, +bounds->thickness() / 2)));
  points.push_back(
      geomDet->toGlobal(LocalPoint(-bounds->width() / 2, -bounds->length() / 2, +bounds->thickness() / 2)));
  points.push_back(
      geomDet->toGlobal(LocalPoint(+bounds->width() / 2, +bounds->length() / 2, -bounds->thickness() / 2)));
  points.push_back(
      geomDet->toGlobal(LocalPoint(-bounds->width() / 2, +bounds->length() / 2, -bounds->thickness() / 2)));
  points.push_back(
      geomDet->toGlobal(LocalPoint(+bounds->width() / 2, -bounds->length() / 2, -bounds->thickness() / 2)));
  points.push_back(
      geomDet->toGlobal(LocalPoint(-bounds->width() / 2, -bounds->length() / 2, -bounds->thickness() / 2)));

  return std::pair<const_iterator, const_iterator>(points.begin(), points.end());
}
