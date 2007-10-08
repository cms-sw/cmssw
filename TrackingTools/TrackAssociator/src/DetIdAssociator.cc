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
// $Id: DetIdAssociator.cc,v 1.18.6.1 2007/10/06 05:50:13 jribnik Exp $
//
//


#include "TrackingTools/TrackAssociator/interface/DetIdAssociator.h"
#include "TrackingTools/TrackAssociator/interface/DetIdInfo.h"
#include <map>

DetIdAssociator::DetIdAssociator():
nPhi_(0),nEta_(0),theMap_(0),theMapIsValid_(false),etaBinSize_(0),ivProp_(0)
{
   maxEta_ = etaBinSize_*nEta_/2;
   minTheta_ = 2*atan(exp(-maxEta_));
}

DetIdAssociator::DetIdAssociator(const int nPhi, const int nEta, const double etaBinSize)
  :nPhi_(nPhi),nEta_(nEta),theMapIsValid_(false),etaBinSize_(etaBinSize),ivProp_(0)
{
   if (nEta_ <= 0 || nPhi_ <= 0) throw cms::Exception("FatalError") << "incorrect look-up map size. Cannot initialize such a map.";
   theMap_ = new std::set<DetId>* [nEta_];
   for (int i=0;i<nEta_;++i) theMap_[i] = new std::set<DetId> [nPhi_];
   maxEta_ = etaBinSize_*nEta_/2;
   minTheta_ = 2*atan(exp(-maxEta_));
}
   
DetIdAssociator::~DetIdAssociator(){
   if (! theMap_) return;
   for(int i=nEta_-1;i>=0;--i) delete [] theMap_[i];
   delete [] theMap_;
}

std::set<DetId> DetIdAssociator::getDetIdsCloseToAPoint(const GlobalPoint& direction,
							const int iN) const
{
   unsigned int n = 0;
   if (iN>0) n = iN;
   return getDetIdsCloseToAPoint(direction,n,n,n,n);
}

std::set<DetId> DetIdAssociator::getDetIdsCloseToAPoint(const GlobalPoint& direction,
							const unsigned int iNEtaPlus,
							const unsigned int iNEtaMinus,
							const unsigned int iNPhiPlus,
							const unsigned int iNPhiMinus) const
{
   std::set<DetId> set;
   check_setup();
   if (! theMapIsValid_ ) throw cms::Exception("FatalError") << "map is not valid.";
   LogTrace("TrackAssociator") << "(iNEtaPlus, iNEtaMinus, iNPhiPlus, iNPhiMinus): " <<
     iNEtaPlus << ", " << iNEtaMinus << ", " << iNPhiPlus << ", " << iNPhiMinus;
   LogTrace("TrackAssociator") << "point (eta,phi): " << direction.eta() << "," << direction.phi();
   int ieta = iEta(direction);
   int iphi = iPhi(direction);
   LogTrace("TrackAssociator") << "(ieta,iphi): " << ieta << "," << iphi << "\n";
   if (ieta>=0 && ieta<nEta_ && iphi>=0 && iphi<nPhi_){
      set = theMap_[ieta][iphi];
      /* 
	for( std::set<DetId>::const_iterator itr=set.begin();
	     itr!=set.end(); itr++)
	  {
	     GlobalPoint point = getPosition(*itr);
	     LogTrace("TrackAssociator") << "\t\tDetId: " <<itr->rawId()<<" \t(eta,phi): " << point.eta() << "," << point.phi() <<std::endl;
	  }
       */
      // dumpMapContent(ieta,iphi);
      // check if any neighbor bin is requested
      if (iNEtaPlus + iNEtaMinus + iNPhiPlus + iNPhiMinus >0 ){
	 LogTrace("TrackAssociator") << "Add neighbors (ieta,iphi): " << ieta << "," << iphi;
	 // eta
	 int maxIEta = ieta+iNEtaPlus;
	 int minIEta = ieta-iNEtaMinus;
	 if (maxIEta>=nEta_) maxIEta = nEta_-1;
	 if (minIEta<0) minIEta = 0;
	 // phi
	 int maxIPhi = iphi+iNPhiPlus;
	 int minIPhi = iphi-iNPhiMinus;
	 if (maxIPhi-minIPhi>=nPhi_){ // all elements in phi
	    minIPhi = 0;
	    maxIPhi = nPhi_-1;
	 }
	 if(minIPhi<0) {
	    minIPhi+=nPhi_;
	    maxIPhi+=nPhi_;
	 }
	 LogTrace("TrackAssociator") << "\tieta (min,max): " << minIEta << "," << maxIEta;
	 LogTrace("TrackAssociator") << "\tiphi (min,max): " << minIPhi << "," << maxIPhi<< "\n";
	 // dumpMapContent(minIEta,maxIEta,minIPhi,maxIPhi);
	 for (int i=minIEta;i<=maxIEta;i++)
	   for (int j=minIPhi;j<=maxIPhi;j++) {
	      // edm::LogVerbatim("TrackAssociator") << "iEta,iPhi,N DetIds: " << i << ", " << j <<
	      // ", " << theMap_[i][j%nPhi_].size();
	      if( i==ieta && j==iphi) continue; // already in the set
	      set.insert((theMap_[i][j%nPhi_]).begin(),(theMap_[i][j%nPhi_]).end());
	   }
      }
      
   }
   return set;
}

std::set<DetId> DetIdAssociator::getDetIdsCloseToAPoint(const GlobalPoint& point,
							const double d) const
{
   return getDetIdsCloseToAPoint(point,d,d,d,d);
}

std::set<DetId> DetIdAssociator::getDetIdsCloseToAPoint(const GlobalPoint& point,
							const double dThetaPlus,
							const double dThetaMinus,
							const double dPhiPlus,
							const double dPhiMinus) const
{
   LogTrace("TrackAssociator") << "(dThetaPlus,dThetaMinus,dPhiPlus,dPhiMinus): " <<
     dThetaPlus << ", " << dThetaMinus << ", " << dPhiPlus << ", " << dPhiMinus;
   unsigned int n = 0;
   if ( dThetaPlus<0 || dThetaMinus<0 || dPhiPlus<0 || dPhiMinus<0) 
     return getDetIdsCloseToAPoint(point,n,n,n,n);
   // check that region of interest overlaps with the look-up map
   double maxTheta = point.theta()+dThetaPlus;
   if (maxTheta > M_PI-minTheta_) maxTheta =  M_PI-minTheta_;
   double minTheta = point.theta()-dThetaMinus;
   if (minTheta < minTheta_) minTheta = minTheta_;
   if ( maxTheta < minTheta_ || minTheta > M_PI-minTheta_) return std::set<DetId>();
   
   // take into account non-linear dependence of eta from
   // theta in regions with large |eta|
   double minEta = -log(tan(maxTheta/2));
   double maxEta = -log(tan(minTheta/2));
   unsigned int iNEtaPlus  = abs(int( ( maxEta-point.eta() )/etaBinSize_));
   unsigned int iNEtaMinus = abs(int( ( point.eta() - minEta )/etaBinSize_));
   unsigned int iNPhiPlus = abs(int( dPhiPlus/(2*M_PI)*nPhi_ ));
   unsigned int iNPhiMinus  = abs(int( dPhiMinus/(2*M_PI)*nPhi_ ));
   // add one more bin in each direction to guaranty that we don't miss anything
   return getDetIdsCloseToAPoint(point, iNEtaPlus+1, iNEtaMinus+1, iNPhiPlus+1, iNPhiMinus+1);
}


int DetIdAssociator::iEta (const GlobalPoint& point) const
{
   return int(point.eta()/etaBinSize_ + nEta_/2);
}

int DetIdAssociator::iPhi (const GlobalPoint& point) const
{
   return int((double(point.phi())+M_PI)/(2*M_PI)*nPhi_);
}


void DetIdAssociator::buildMap()
{
   check_setup();
   LogTrace("TrackAssociator")<<"building map" << "\n";
   // clear the map
   if (nEta_ <= 0 || nPhi_ <= 0) throw cms::Exception("FatalError") << "incorrect look-up map size. Cannot build such a map.";
   if (! theMap_) throw cms::Exception("FatalError") << "incorrect look-up map. Cannot build such a map.";
   for(int i=0;i<nEta_;++i)
     for(int j=0;j<nPhi_;++j)
       theMap_[i][j].clear();
   int numberOfDetIdsOutsideEtaRange = 0;
   int numberOfDetIdsActive = 0;
   std::set<DetId> validIds = getASetOfValidDetIds();
   LogTrace("TrackAssociator")<< "Number of valid DetIds: " <<  validIds.size();
   for (std::set<DetId>::const_iterator id_itr = validIds.begin(); id_itr!=validIds.end(); id_itr++) {
      std::vector<GlobalPoint> points = getDetIdPoints(*id_itr);
      LogTrace("TrackAssociatorVerbose")<< "Found " << points.size() << " global points to describe geometry of DetId: " 
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
	      edm::LogWarning("TrackAssociator") << "Critical error! Bad detector unit geometry:\n\tDetId:" 
		<< id_itr->rawId() << "\t mag(): " << iter->mag() << "\n" << DetIdInfo::info( *id_itr )
		  << "\nSkipped the element";
	      continue;
	   }
	   volume_.addActivePoint(*iter);
	   int ieta = iEta(*iter);
	   int iphi = iPhi(*iter);
	   if (ieta<0 || ieta>=nEta_) {
	      LogTrace("TrackAssociator")<<"Out of range: DetId:" << id_itr->rawId() << "\t (ieta,iphi): " 
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
	 LogTrace("TrackAssociator")<<"Out of range or no geometry: DetId:" << id_itr->rawId() <<
	   "\n\teta (min,max): " << etaMin << "," << etaMax <<
	   "\n\tphi (min,max): " << phiMin << "," << phiMax <<
	   "\nTower id: " << id_itr->rawId() << "\n";
	 numberOfDetIdsOutsideEtaRange++;
	 continue;
      }
	  
      LogTrace("TrackAssociatorVerbose") << "DetId (ieta_min,ieta_max,iphi_min,iphi_max): " << id_itr->rawId() <<
	", " << etaMin << ", " << etaMax << ", " << phiMin << ", " << phiMax;
      for(int ieta = etaMin; ieta <= etaMax; ieta++)
	for(int iphi = phiMin; iphi <= phiMax; iphi++)
	  theMap_[ieta][iphi%nPhi_].insert(*id_itr);
      numberOfDetIdsActive++;
   }
   LogTrace("TrackAssociator") << "Number of elements outside the allowed range ( |eta|>"<<
     nEta_/2*etaBinSize_ << "): " << numberOfDetIdsOutsideEtaRange << "\n";
   LogTrace("TrackAssociator") << "Number of active DetId's mapped: " << 
     numberOfDetIdsActive << "\n";
   volume_.determinInnerDimensions();
   edm::LogVerbatim("TrackAssociator") << "Volume (minR, maxR, minZ, maxZ): " << volume_.minR() << ", " << volume_.maxR() <<
     ", " << volume_.minZ() << ", " << volume_.maxZ();
   theMapIsValid_ = true;
}

std::set<DetId> DetIdAssociator::getDetIdsInACone(const std::set<DetId>& inset, 
					     const std::vector<GlobalPoint>& trajectory,
					     const double dR) const
{
   if ( dR > 2*M_PI && dR > maxEta_ ) return inset;
   check_setup();
   std::set<DetId> outset;
   for(std::set<DetId>::const_iterator id_iter = inset.begin(); id_iter != inset.end(); id_iter++)
     for(std::vector<GlobalPoint>::const_iterator point_iter = trajectory.begin(); point_iter != trajectory.end(); point_iter++)
       if (nearElement(*point_iter,*id_iter,dR)) {
	  outset.insert(*id_iter);
	  break;
       }
   return outset;
}

std::set<DetId> DetIdAssociator::getCrossedDetIds(const std::set<DetId>& inset,
						  const std::vector<GlobalPoint>& trajectory) const
{
   check_setup();
   std::set<DetId> outset;
   for(std::set<DetId>::const_iterator id_iter = inset.begin(); id_iter != inset.end(); id_iter++) 
     for(std::vector<GlobalPoint>::const_iterator point_iter = trajectory.begin(); point_iter != trajectory.end(); point_iter++)
       if (insideElement(*point_iter, *id_iter))  {
	  outset.insert(*id_iter);
	  break;
       }
   return outset;
}

std::vector<DetId> DetIdAssociator::getCrossedDetIdsOrdered(const std::set<DetId>& inset,
							   const std::vector<GlobalPoint>& trajectory) const
{
   check_setup();
   std::vector<DetId> output;
   std::set<DetId> ids(inset);
   for(std::vector<GlobalPoint>::const_iterator point_iter = trajectory.begin(); 
       point_iter != trajectory.end(); point_iter++)
     {
	std::set<DetId>::const_iterator id_iter = ids.begin();
	while ( id_iter != ids.end() ) {
	   if (insideElement(*point_iter, *id_iter)) {
	      output.push_back(*id_iter);
	      ids.erase(id_iter++);
	   }else
	     id_iter++;
	}
     }
   return output;
}

void DetIdAssociator::dumpMapContent(int ieta, int iphi) const
{
   if (! (ieta>=0 && ieta<nEta_ && iphi>=0) )
     {
	edm::LogWarning("TrackAssociator") << "ieta or iphi is out of range. Skipped.";
	return;
     }

   std::set<DetId> set = theMap_[ieta][iphi%nPhi_];
   LogTrace("TrackAssociator") << "Map content for cell (ieta,iphi): " << ieta << ", " << iphi%nPhi_;
   for(std::set<DetId>::const_iterator itr = set.begin(); itr!=set.end(); itr++)
     {
	LogTrace("TrackAssociator") << "\tDetId " << itr->rawId() << ", geometry (x,y,z,rho,eta,phi):";
	std::vector<GlobalPoint> points = getDetIdPoints(*itr);
	for(std::vector<GlobalPoint>::const_iterator point = points.begin(); point != points.end(); point++)
	  LogTrace("TrackAssociator") << "\t\t" << point->x() << ", " << point->y() << ", " << point->z() << ", "
	  << point->perp() << ", " << point->eta() << ", " << point->phi();
     }
}

void DetIdAssociator::dumpMapContent(int ieta_min, int ieta_max, int iphi_min, int iphi_max) const
{
   for(int i=ieta_min;i<=ieta_max;i++)
     for(int j=iphi_min;j<=iphi_max;j++)
       dumpMapContent(i,j);
}

const FiducialVolume& DetIdAssociator::volume() const
{
   if (! theMapIsValid_ ) throw cms::Exception("FatalError") << "map is not valid.";
   return volume_; 
}

std::set<DetId> DetIdAssociator::getDetIdsCloseToAPoint(const GlobalPoint& direction,
							const MapRange& mapRange) const
{
   return getDetIdsCloseToAPoint(direction, mapRange.dThetaPlus, mapRange.dThetaMinus,
				 mapRange.dPhiPlus, mapRange.dPhiMinus);

}

