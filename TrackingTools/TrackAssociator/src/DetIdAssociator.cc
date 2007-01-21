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
// $Id: DetIdAssociator.cc,v 1.8 2007/01/20 17:29:48 dmytro Exp $
//
//


#include "TrackingTools/TrackAssociator/interface/DetIdAssociator.h"
#include <map>

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
      // dumpMapContent(ieta,iphi);
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
	 // dumpMapContent(minIEta,maxIEta,minIPhi,maxIPhi);
	 for (int i=minIEta;i<=maxIEta;i++)
	   for (int j=minIPhi;j<=maxIPhi;j++) {
	      if( i==ieta && j==iphi) continue; // already in the set
	      set.insert((*theMap_)[i][j%nPhi_].begin(),(*theMap_)[i][j%nPhi_].end());
	   }
      }
      
   }
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
   LogTrace("DetIdAssociator")<< "Number of valid DetIds: " <<  validIds.size();
   for (std::set<DetId>::const_iterator id_itr = validIds.begin(); id_itr!=validIds.end(); id_itr++) {
      std::vector<GlobalPoint> points = getDetIdPoints(*id_itr);
      LogTrace("DetIdAssociator")<< "Found " << points.size() << " global points to describe geometry of DetId: " 
	<< id_itr->rawId();
      int etaMax(-1);
      int etaMin(-1);
      int phiMax(-1);
      int phiMin(-1);
      // this is a bit overkill, but it should be 100% proof (when debugged :)
      for(std::vector<GlobalPoint>::const_iterator iter = points.begin(); iter != points.end(); iter++)
	{
	   // FIX ME: this should be a fatal error
	   if(isnan(iter->mag())||iter->mag()>1e5) { //Detector parts cannot be 1 km away or be NaN
	      edm::LogWarning("DetIdAssociator") << "Critical error! Bad detector unit geometry:\n\tDetId:" <<
		id_itr->rawId() << "\t mag(): " << iter->mag() << "\nSkipped the element";
	      continue;
	   }
	   int ieta = iEta(*iter);
	   int iphi = iPhi(*iter);
	   if (ieta<0 || ieta>=nEta_) {
	      LogTrace("DetIdAssociator")<<"Out of range: DetId:" << id_itr->rawId() << "\t (ieta,iphi): " 
		<< ieta << "," << iphi << "\n" << "Point: " << *iter << "\t(eta,phi): " << (*iter).eta() 
		  << "," << (*iter).phi() << "\n center: " << getPosition(*id_itr);
	      continue;
	   }
	   if ( phiMin<0 ) {
		// first element
		etaMin = ieta;
	        etaMax = ieta;
	        phiMin = iphi;
	        phiMax = iphi;
	   }else{
	      // check for discontinuity in phi
	      int deltaMin = abs(phiMin -iphi);
	      int deltaMax = abs(phiMax -iphi);
	      // assume that no single detector element has more than 3.1416 coverage in phi
	      if ( deltaMin > nPhi_/2 && phiMin < nPhi_/2 ) phiMin+= nPhi_;
	      if ( deltaMax > nPhi_/2 ) {
		 if (phiMax < nPhi_/2 ) 
		   phiMax+= nPhi_;
		 else
		   iphi += nPhi_;
	      }
	      assert (iphi>=0);
	      if ( etaMin > ieta) etaMin = ieta;
	      if ( etaMax < ieta) etaMax = ieta;
	      if ( phiMin > iphi) phiMin = iphi;
	      if ( phiMax < iphi) phiMax = iphi;
	   }
	}
      if (etaMax<0||phiMax<0||etaMin>=nEta_||phiMin>=nPhi_) {
	 LogTrace("DetIdAssociator")<<"Out of range: DetId:" << id_itr->rawId() <<
	   "\n\teta (min,max): " << etaMin << "," << etaMax <<
	   "\n\tphi (min,max): " << phiMin << "," << phiMax <<
	   "\nTower id: " << id_itr->rawId() << "\n";
	 numberOfDetIdsOutsideEtaRange++;
	 continue;
      }
	  
      LogTrace("") << "DetId (ieta_min,ieta_max,iphi_min,iphi_max): " << id_itr->rawId() <<
	", " << etaMin << ", " << etaMax << ", " << phiMin << ", " << phiMax;
      for(int ieta = etaMin; ieta <= etaMax; ieta++)
	for(int iphi = phiMin; iphi <= phiMax; iphi++)
	  (*theMap_)[ieta][iphi%nPhi_].insert(*id_itr);
      numberOfDetIdsActive++;
   }
   LogTrace("DetIdAssociator") << "Number of elements outside the allowed range ( |eta|>"<<
     nEta_/2*etaBinSize_ << "): " << numberOfDetIdsOutsideEtaRange << "\n";
   LogTrace("DetIdAssociator") << "Number of active DetId's mapped: " << 
     numberOfDetIdsActive << "\n";
}

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

void DetIdAssociator::dumpMapContent(int ieta, int iphi)
{
   if (! (ieta>=0 && ieta<nEta_ && iphi>=0) )
     {
	edm::LogWarning("BadRequest") << "ieta or iphi is out of range. Skipped.";
	return;
     }

   std::set<DetId> set = (*theMap_)[ieta][iphi%nPhi_];
   LogTrace("") << "Map content for cell (ieta,iphi): " << ieta << ", " << iphi%nPhi_;
   for(std::set<DetId>::const_iterator itr = set.begin(); itr!=set.end(); itr++)
     {
	LogTrace("") << "\tDetId " << itr->rawId() << ", geometry (x,y,z,rho,eta,phi):";
	std::vector<GlobalPoint> points = getDetIdPoints(*itr);
	for(std::vector<GlobalPoint>::const_iterator point = points.begin(); point != points.end(); point++)
	  LogTrace("") << "\t\t" << point->x() << ", " << point->y() << ", " << point->z() << ", "
	  << point->perp() << ", " << point->eta() << ", " << point->phi();
     }
}

void DetIdAssociator::dumpMapContent(int ieta_min, int ieta_max, int iphi_min, int iphi_max)
{
   for(int i=ieta_min;i<=ieta_max;i++)
     for(int j=iphi_min;j<=iphi_max;j++)
       dumpMapContent(i,j);
}


