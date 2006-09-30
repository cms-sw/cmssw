// -*- C++ -*-
//
// Package:    TrackAssociator
// Class:      DetIdAssociator
// 
/*

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Dmytro Kovalskyi
//         Created:  Fri Apr 21 10:59:41 PDT 2006
// $Id: DetIdAssociator.cc,v 1.3 2006/08/15 23:03:53 dmytro Exp $
//
//


#include "TrackingTools/TrackAssociator/interface/DetIdAssociator.h"
#include <map>


// surfaces is a vector of GlobalPoint representing outermost point on a cylinder
std::vector<GlobalPoint> DetIdAssociator::getTrajectory( const FreeTrajectoryState& ftsStart,
							 const std::vector<GlobalPoint>& surfaces,
							 const double etaOverlap)
{
   check_setup();
   std::vector<GlobalPoint> trajectory;
   TrajectoryStateOnSurface tSOSDest;
   FreeTrajectoryState ftsCurrent = ftsStart;

   for(std::vector<GlobalPoint>::const_iterator surface_iter = surfaces.begin(); 
       surface_iter != surfaces.end(); surface_iter++) {
      PropagationTarget target;
      // define limiting surfaces using some stuff, which is some
      // weird pointer owning the object, so no need to delete objects
      std::map<PropagationTarget, Surface*> map;
      map[Barrel] = new Cylinder(Surface::PositionType(0,0,0),
				 Surface::RotationType(), 
				 double (surface_iter->perp()) );
      map[ForwardEndcap] = new Plane(Surface::PositionType(0,0,surface_iter->z()),
				     Surface::RotationType());
      map[BackwardEndcap] = new Plane(Surface::PositionType(0,0,-surface_iter->z()),
				      Surface::RotationType());
      
      LogTrace("StartingPoint")<< "Propagate from "<< "\n"
	<< "\tx: " << ftsStart.position().x()<< "\n"
	<< "\ty: " << ftsStart.position().y()<< "\n"
	<< "\tz: " << ftsStart.position().z()<< "\n"
	<< "\tmomentum eta: " << ftsStart.momentum().eta()<< "\n"
	<< "\tmomentum phi: " << ftsStart.momentum().phi()<< "\n"
	<< "\tmomentum: " << ftsStart.momentum().mag()<< "\n";
      
      // First propage the track to the cylinder if |eta|<1, othewise to the encap
      // and correct depending on the result
      if (fabs(ftsCurrent.momentum().eta())<1)
	target = Barrel;
      else {
	 if(ftsCurrent.momentum().eta()>1)
	   target = ForwardEndcap;
	 else
	   target = BackwardEndcap;
      }
      
      tSOSDest = ivProp_->propagate(ftsCurrent, *map[target]);
      GlobalPoint point = tSOSDest.freeState()->position();

      // if near the edge
      if ( fabs(fabs(point.eta())-fabs(surface_iter->eta()))<etaOverlap ) {
	 trajectory.push_back(point);
	 if (target != Barrel)
	   target = Barrel;
	 else {
	    if(ftsCurrent.momentum().eta()>0)
	      target = ForwardEndcap;
	    else
	      target = BackwardEndcap;
	 }
      } else {
	 // If missed the target, propagate to other targets.
	 PropagationTarget newTarget = target;
	 if (point.perp() > surface_iter->perp())
	   newTarget = Barrel; 
	 if (point.z() > surface_iter->z())
	   target = ForwardEndcap;
	 if (point.z() < -surface_iter->z())
	   target = BackwardEndcap;
	 
	 if(newTarget == target) {
	    ftsCurrent = *tSOSDest.freeState();
	    trajectory.push_back(point);
	    continue;
	 }
	 target = newTarget;
      }
      
      tSOSDest = ivProp_->propagate(ftsStart, *map[target]);
      if (! tSOSDest.isValid()) throw cms::Exception("FatalError") << "Failed to propagate the track\n";
      point = tSOSDest.freeState()->position();
      
      LogTrace("SuccessfullPropagation") << "Great, I reached something." << "\n"
	<< "\tx: " << tSOSDest.freeState()->position().x() << "\n"
	<< "\ty: " << tSOSDest.freeState()->position().y() << "\n"
	<< "\tz: " << tSOSDest.freeState()->position().z() << "\n"
	<< "\teta: " << tSOSDest.freeState()->position().eta() << "\n"
	<< "\tphi: " << tSOSDest.freeState()->position().phi() << "\n";
      
      trajectory.push_back(point);
   }
   return trajectory;
}

std::set<DetId> DetIdAssociator::getDetIdsCloseToAPoint(const GlobalPoint& direction,
						   const int idR)
{
   std::set<DetId> set;
   check_setup();
   if (! theMap_) buildMap();
   LogTrace("MatchPoint") << "point (eta,phi): " << direction.eta() << "," << direction.phi() << "\n";
   int ieta = iEta(direction);
   int iphi = iPhi(direction);
   LogTrace("MatchPoint") << "(ieta,iphi): " << ieta << "," << iphi << "\n";
   if (ieta>=0 && ieta<nEta_ && iphi>=0 && iphi<nPhi_){
      set = (*theMap_)[ieta][iphi];
      /*      if (debug_>1)
	for( std::set<DetId>::const_iterator itr=set.begin();
	     itr!=set.end(); itr++)
	  {
	     GlobalPoint point = getPosition(*itr);
	     LogTrace("MatchPoint") << "\t\tDetId: " <<itr->rawId()<<" \t(eta,phi): " << point.eta() << "," << point.phi() <<std::endl;
	  }
       */    
      if (idR>0){
	  LogTrace("MatchPoint") << "Add neighbors (ieta,iphi): " << ieta << "," << iphi << "\n";
	 //add neighbors
	 int maxIEta = ieta+idR;
	 int minIEta = ieta-idR;
	 if(maxIEta>=nEta_) maxIEta = nEta_-1;
	 if(minIEta<0) minIEta = 0;
	 int maxIPhi = iphi+idR;
	 int minIPhi = iphi-idR;
	 if(minIPhi<0) {
	    minIPhi+=nPhi_;
	    maxIPhi+=nPhi_;
	 }
	 LogTrace("MatchPoint") << "\tieta (min,max): " << minIEta << "," << maxIEta<< "\n";
	 LogTrace("MatchPoint") << "\tiphi (min,max): " << minIPhi << "," << maxIPhi<< "\n";
	 for (int i=minIEta;i<=maxIEta;i++)
	   for (int j=minIPhi;j<=maxIPhi;j++) {
	      if( i==ieta && j==iphi) continue; // already in the set
	      set.insert((*theMap_)[i][j%nPhi_].begin(),(*theMap_)[i][j%nPhi_].end());
	   }
      }
   }
 /*  if (debug_)
     for( std::set<DetId>::const_iterator itr=set.begin();
	  itr!= set.end(); itr++)
       {
	  GlobalPoint point = getPosition(*itr);
	  std::cout << "\t\tDetId: " <<itr->rawId() <<" \t(eta,phi): " << point.eta() << "," << point.phi() <<std::endl;
       }
  */
   return set;
}

int DetIdAssociator::iEta (const GlobalPoint& point)
{
   return int(point.eta()/etaBinSize_ + nEta_/2);
}

int DetIdAssociator::iPhi (const GlobalPoint& point)
{
   return int((double(point.phi())+3.1416)/(2*3.1416)*nPhi_);
}


void DetIdAssociator::buildMap()
{
   check_setup();
   LogTrace("DetIdAssociator")<<"building map" << "\n";
   if(theMap_) delete theMap_;
   theMap_ = new std::vector<std::vector<std::set<DetId> > >(nEta_,nPhi_);
   int numberOfDetIdsOutsideEtaRange = 0;
   int numberOfDetIdsActive = 0;
   std::set<DetId> validIds = getASetOfValidDetIds();
   for (std::set<DetId>::const_iterator id_itr = validIds.begin(); id_itr!=validIds.end(); id_itr++) {	 
      std::vector<GlobalPoint> points = getDetIdPoints(*id_itr);
      int etaMax(-1);
      int etaMin(nEta_);
      int phiMax(-1);
      int phiMin(nPhi_);
      // this is a bit overkill, but it should be 100% proof (when debugged :)
      for(std::vector<GlobalPoint>::const_iterator iter = points.begin(); iter != points.end(); iter++)
	{
	   int ieta = iEta(*iter);
	   int iphi = iPhi(*iter);
	   if (ieta<0 || ieta>=nEta_) {
	      LogTrace("DetIdAssociator")<<"Out of range: DetId:" << id_itr->rawId() << "\t (ieta,iphi): " 
		<< ieta << "," << iphi << "\n" << "Point: " << *iter << "\t(eta,phi): " << (*iter).eta() 
		  << "," << (*iter).phi() << "\n center: " << getPosition(*id_itr) <<"\n";
	      continue;
	   }
	   if ( iphi >= nPhi_ ) iphi = iphi % nPhi_;
	   assert (iphi>=0);
	   if ( etaMin > ieta) etaMin = ieta;
	   if ( etaMax < ieta) etaMax = ieta;
	   if ( phiMin > iphi) phiMin = iphi;
	   if ( phiMax < iphi) phiMax = iphi;
	}
      if (etaMax<0||phiMax<0||etaMin>=nEta_||phiMin>=nPhi_) {
	 LogTrace("DetIdAssociator")<<"Out of range: DetId:" << id_itr->rawId() <<
	   "\n\teta (min,max): " << etaMin << "," << etaMax <<
	   "\n\tphi (min,max): " << phiMin << "," << phiMax <<
	   "\nTower id: " << id_itr->rawId() << "\n";
	 numberOfDetIdsOutsideEtaRange++;
	 continue;
      }
	  
      if (phiMax-phiMin > phiMin+nPhi_-phiMax){
	 // found discontinuity in phi, make phi continues
	 phiMin += nPhi_;
	 std::swap(phiMin,phiMax);
      }
      for(int ieta = etaMin; ieta <= etaMax; ieta++)
	for(int iphi = phiMin; iphi <= phiMax; iphi++)
	  (*theMap_)[ieta][iphi%nPhi_].insert(*id_itr);
      numberOfDetIdsActive++;
   }
   LogTrace("DetIdAssociator") << "Number of elements outside the allowed range ( |eta|>"<<
     nEta_/2*etaBinSize_ << "): " << numberOfDetIdsOutsideEtaRange << "\n";
   LogTrace("DetIdAssociator") << "Number of active DetId's mapped: " << 
     numberOfDetIdsActive << "\n";
/*   if (debug_){
      std::cout << "The map:" << std::endl;
      for (int i=0;i<nEta_;i++)
	for (int j=0;j<nPhi_;j++)
	  {
	     std::cout <<"\t" << i << "," <<j << "\t number of elements:" <<(*theMap_)[i][j].size() <<std::endl;
	     for( std::set<DetId>::const_iterator itr=(*theMap_)[i][j].begin();
		  itr!= (*theMap_)[i][j].end(); itr++)
	       {
		  GlobalPoint point = getPosition(*itr);
		  std::cout << "\t\tDetId: " <<itr->rawId() <<" \t(eta,phi): " << point.eta() << "," << point.phi() <<std::endl;
	       }
	  }
        } 
  */ 
}

/*
 	 if (f1() == 0 ) 
	  {
	     
	         string errorMessage = "Input file " + colliFile + "not found";
	     edm::LogError("CSCGasCollisions") << errorMessage << " in path " << path
	                 << "\nSet Muon:Endcap:CollisionsFile in .orcarc to the "
	                " location of the file relative to ORCA_DATA_PATH." ;
	     throw cms::Exception( " Endcap Muon gas collisions data file not found.");
	  }
	
*/

std::set<DetId> DetIdAssociator::getDetIdsInACone(const std::set<DetId>& inset, 
					     const std::vector<GlobalPoint>& trajectory,
					     const double dR)
{
   check_setup();
   std::set<DetId> outset;
   for(std::set<DetId>::const_iterator id_iter = inset.begin(); id_iter != inset.end(); id_iter++)
     for(std::vector<GlobalPoint>::const_iterator point_iter = trajectory.begin(); point_iter != trajectory.end(); point_iter++)
       if (nearElement(*point_iter,*id_iter,dR)) outset.insert(*id_iter);
   return outset;
}

std::set<DetId> DetIdAssociator::getCrossedDetIds(const std::set<DetId>& inset,
					     const std::vector<GlobalPoint>& trajectory)
{
   check_setup();
   std::set<DetId> outset;
   for(std::set<DetId>::const_iterator id_iter = inset.begin(); id_iter != inset.end(); id_iter++) 
     for(std::vector<GlobalPoint>::const_iterator point_iter = trajectory.begin(); point_iter != trajectory.end(); point_iter++)
       if (insideElement(*point_iter, *id_iter))  outset.insert(*id_iter);
   return outset;
}

