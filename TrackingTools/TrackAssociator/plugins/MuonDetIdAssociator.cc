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
// $Id: MuonDetIdAssociator.cc,v 1.1 2011/04/07 09:12:02 innocent Exp $
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

void MuonDetIdAssociator::check_setup() const {
   if (geometry_==0) throw cms::Exception("ConfigurationProblem") << "GlobalTrackingGeomtry is not set\n";
   if (cscbadchambers_==0) throw cms::Exception("ConfigurationProblem") << "CSCBadChambers is not set\n";
   DetIdAssociator::check_setup();
}

const GeomDet* MuonDetIdAssociator::getGeomDet( const DetId& id ) const
{
   if (geometry_==0) throw cms::Exception("ConfigurationProblem") << "GlobalTrackingGeomtry is not set\n";
   const GeomDet* gd = geometry_->idToDet(id);
   if (gd == 0) throw cms::Exception("NoGeometry") << "Cannot find GeomDet for DetID: " << id.rawId() <<"\n";
   return gd;
}


GlobalPoint MuonDetIdAssociator::getPosition(const DetId& id) const {
   Surface::PositionType point(getGeomDet(id)->surface().position());
   return GlobalPoint(point.x(),point.y(),point.z());
}

const std::vector<DetId>& MuonDetIdAssociator::getValidDetIds(unsigned int subDectorIndex) const {
  validIds_.clear();
  if (geometry_==0) throw cms::Exception("ConfigurationProblem") << "GlobalTrackingGeomtry is not set\n";
  if (subDectorIndex!=0) throw cms::Exception("FatalError") << 
    "Muon sub-dectors are all handle as one sub-system, but subDetectorIndex is not zero.\n";

  // CSC 
  if (! geometry_->slaveGeometry(CSCDetId()) ) throw cms::Exception("FatalError") << "Cannnot CSCGeometry\n";
  const std::vector<GeomDet*>& geomDetsCSC = geometry_->slaveGeometry(CSCDetId())->dets();
  for(std::vector<GeomDet*>::const_iterator it = geomDetsCSC.begin(); it != geomDetsCSC.end(); ++it)
    if (CSCChamber* csc = dynamic_cast< CSCChamber*>(*it)) {
      if ((! includeBadChambers_) && (cscbadchambers_->isInBadChamber(CSCDetId(csc->id())))) continue;
      validIds_.push_back(csc->id());
    }
  
  // DT
  if (! geometry_->slaveGeometry(DTChamberId()) ) throw cms::Exception("FatalError") << "Cannnot DTGeometry\n";
  const std::vector<GeomDet*>& geomDetsDT = geometry_->slaveGeometry(DTChamberId())->dets();
  for(std::vector<GeomDet*>::const_iterator it = geomDetsDT.begin(); it != geomDetsDT.end(); ++it)
    if (DTChamber* dt = dynamic_cast< DTChamber*>(*it)) validIds_.push_back(dt->id());

  // RPC
  if (! geometry_->slaveGeometry(RPCDetId()) ) throw cms::Exception("FatalError") << "Cannnot RPCGeometry\n";
  const std::vector<GeomDet*>& geomDetsRPC = geometry_->slaveGeometry(RPCDetId())->dets();
  for(std::vector<GeomDet*>::const_iterator it = geomDetsRPC.begin(); it != geomDetsRPC.end(); ++it)
    if (RPCChamber* rpc = dynamic_cast< RPCChamber*>(*it)) validIds_.push_back(rpc->id());

  return validIds_;
}

bool MuonDetIdAssociator::insideElement(const GlobalPoint& point, const DetId& id) const {
   if (geometry_==0) throw cms::Exception("ConfigurationProblem") << "GlobalTrackingGeomtry is not set\n";
   LocalPoint lp = geometry_->idToDet(id)->toLocal(point);
   return geometry_->idToDet(id)->surface().bounds().inside(lp);
}

std::pair<DetIdAssociator::const_iterator,DetIdAssociator::const_iterator> 
MuonDetIdAssociator::getDetIdPoints(const DetId& id) const 
{
   points_.clear();
   const GeomDet* geomDet = getGeomDet( id );
   
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
   points_.push_back(geomDet->toGlobal(LocalPoint(+bounds->width()/2,+bounds->length()/2,+bounds->thickness()/2)));
   points_.push_back(geomDet->toGlobal(LocalPoint(-bounds->width()/2,+bounds->length()/2,+bounds->thickness()/2)));
   points_.push_back(geomDet->toGlobal(LocalPoint(+bounds->width()/2,-bounds->length()/2,+bounds->thickness()/2)));
   points_.push_back(geomDet->toGlobal(LocalPoint(-bounds->width()/2,-bounds->length()/2,+bounds->thickness()/2)));
   points_.push_back(geomDet->toGlobal(LocalPoint(+bounds->width()/2,+bounds->length()/2,-bounds->thickness()/2)));
   points_.push_back(geomDet->toGlobal(LocalPoint(-bounds->width()/2,+bounds->length()/2,-bounds->thickness()/2)));
   points_.push_back(geomDet->toGlobal(LocalPoint(+bounds->width()/2,-bounds->length()/2,-bounds->thickness()/2)));
   points_.push_back(geomDet->toGlobal(LocalPoint(-bounds->width()/2,-bounds->length()/2,-bounds->thickness()/2)));
   
   return std::pair<const_iterator,const_iterator>(points_.begin(),points_.end());
}

void MuonDetIdAssociator::setGeometry(const DetIdAssociatorRecord& iRecord)
{
  edm::ESHandle<GlobalTrackingGeometry> geometryH;
  iRecord.getRecord<GlobalTrackingGeometryRecord>().get(geometryH);
  setGeometry(geometryH.product());
}
