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
// $Id: MuonDetIdAssociator.cc,v 1.8 2008/03/31 13:31:42 dmytro Exp $
//
//


#include "TrackingTools/TrackAssociator/interface/MuonDetIdAssociator.h"
// #include "Utilities/Timing/interface/TimerStack.h"
#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixPropagator.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "Geometry/DTGeometry/interface/DTChamber.h"
#include "Geometry/CSCGeometry/interface/CSCChamber.h"
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

std::set<DetId> MuonDetIdAssociator::getASetOfValidDetIds() const {
   std::set<DetId> setOfValidIds;
   if (geometry_==0) throw cms::Exception("ConfigurationProblem") << "GlobalTrackingGeomtry is not set\n";
   // we need to store only DTChambers as well as CSCChambers
   // Let's get all GeomDet by dets() and select only DTChambers and CSCChambers

   // comment this for now, till the fix of GlobalTrackingGeometry is in a release
   // std::vector<GeomDet*> vectOfGeomDetPtrs = geometry_->dets();
   // LogTrace("TrackAssociator") << "Number of GeomDet found: " << vectOfGeomDetPtrs.size();
   // for(std::vector<GeomDet*>::const_iterator it = vectOfGeomDetPtrs.begin(); it != vectOfGeomDetPtrs.end(); ++it)
   //  {
   // 	if ((*it)->subDetector() == GeomDetEnumerators::CSC || (*it)->subDetector() == GeomDetEnumerators::DT)
   //	  {
   //	     if (DTChamber* dt = dynamic_cast< DTChamber*>(*it)) {
   //		setOfValidIds.insert(dt->id());
   //	     }else{
   //		if (CSCChamber* csc = dynamic_cast< CSCChamber*>(*it)) {
   //		   setOfValidIds.insert(csc->id());
   //		}
   //	     }
   //	  }
   //   }
   
   // CSC 
   if (! geometry_->slaveGeometry(CSCDetId()) ) throw cms::Exception("FatalError") << "Cannnot CSCGeometry\n";
   std::vector<GeomDet*> geomDetsCSC = geometry_->slaveGeometry(CSCDetId())->dets();
   for(std::vector<GeomDet*>::const_iterator it = geomDetsCSC.begin(); it != geomDetsCSC.end(); ++it)
     if (CSCChamber* csc = dynamic_cast< CSCChamber*>(*it)) {
       if ((! includeBadChambers_) && (isBadCSCChamber(CSCDetId(csc->id())))) continue;
       setOfValidIds.insert(csc->id());
     }
   
   // DT
   if (! geometry_->slaveGeometry(DTChamberId()) ) throw cms::Exception("FatalError") << "Cannnot DTGeometry\n";
   std::vector<GeomDet*> geomDetsDT = geometry_->slaveGeometry(DTChamberId())->dets();
   for(std::vector<GeomDet*>::const_iterator it = geomDetsDT.begin(); it != geomDetsDT.end(); ++it)
     if (DTChamber* dt = dynamic_cast< DTChamber*>(*it)) setOfValidIds.insert(dt->id());

   return setOfValidIds;
}

bool MuonDetIdAssociator::insideElement(const GlobalPoint& point, const DetId& id) const {
   if (geometry_==0) throw cms::Exception("ConfigurationProblem") << "GlobalTrackingGeomtry is not set\n";
   LocalPoint lp = geometry_->idToDet(id)->toLocal(point);
   return geometry_->idToDet(id)->surface().bounds().inside(lp);
}

std::vector<GlobalPoint> MuonDetIdAssociator::getDetIdPoints(const DetId& id) const {
   std::vector<GlobalPoint> points;
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
   points.push_back(geomDet->toGlobal(LocalPoint(+bounds->width()/2,+bounds->length()/2,+bounds->thickness()/2)));
   points.push_back(geomDet->toGlobal(LocalPoint(-bounds->width()/2,+bounds->length()/2,+bounds->thickness()/2)));
   points.push_back(geomDet->toGlobal(LocalPoint(+bounds->width()/2,-bounds->length()/2,+bounds->thickness()/2)));
   points.push_back(geomDet->toGlobal(LocalPoint(-bounds->width()/2,-bounds->length()/2,+bounds->thickness()/2)));
   points.push_back(geomDet->toGlobal(LocalPoint(+bounds->width()/2,+bounds->length()/2,-bounds->thickness()/2)));
   points.push_back(geomDet->toGlobal(LocalPoint(-bounds->width()/2,+bounds->length()/2,-bounds->thickness()/2)));
   points.push_back(geomDet->toGlobal(LocalPoint(+bounds->width()/2,-bounds->length()/2,-bounds->thickness()/2)));
   points.push_back(geomDet->toGlobal(LocalPoint(-bounds->width()/2,-bounds->length()/2,-bounds->thickness()/2)));
   
   return  points;
}
