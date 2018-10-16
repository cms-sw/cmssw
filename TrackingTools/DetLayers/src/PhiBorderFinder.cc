#include "TrackingTools/DetLayers/interface/PhiBorderFinder.h"

PhiBorderFinder::PhiBorderFinder(const std::vector<const Det*>& utheDets) 
  : theNbins(utheDets.size()), 
    isPhiPeriodic_(false), 
    isPhiOverlapping_(false) {
  std::vector<const Det*> theDets = utheDets;
  precomputed_value_sort(theDets.begin(), theDets.end(), DetPhi());
  
  const std::string metname = "Muon|RecoMuon|RecoMuonDetLayers|PhiBorderFinder";
  
  double step = 2.*Geom::pi()/theNbins;
  
  LogTrace(metname) << "RecoMuonDetLayers::PhiBorderFinder "
		    << "step w: " << step << " # of bins: " << theNbins;
  
  std::vector<double> spread(theNbins);
  std::vector<std::pair<double,double> > phiEdge;
  phiEdge.reserve(theNbins);
  thePhiBorders.reserve(theNbins);
  thePhiBins.reserve(theNbins);
  for ( unsigned int i = 0; i < theNbins; i++ ) {
    thePhiBins.push_back(theDets[i]->position().phi());
    spread.push_back(theDets[i]->position().phi()
		     - (theDets[0]->position().phi() + i*step));
    
    LogTrace(metname) << "bin: " << i << " phi bin: " << thePhiBins[i]
		      << " spread: " <<  spread[i];
    
    
    ConstReferenceCountingPointer<BoundPlane> plane = 
      dynamic_cast<const BoundPlane*>(&theDets[i]->surface());
    if (plane==nullptr) {
      //FIXME
      throw cms::Exception("UnexpectedState") << ("PhiBorderFinder: det surface is not a BoundPlane");
    }
    
    std::vector<GlobalPoint> dc = 
      BoundingBox().corners(*plane);
    
    float phimin(999.), phimax(-999.);
    for (std::vector<GlobalPoint>::const_iterator pt=dc.begin();
	 pt!=dc.end(); pt++) {
      float phi = (*pt).phi();
      //	float z = pt->z();
      if (phi < phimin) phimin = phi;
      if (phi > phimax) phimax = phi;
    }
    if (phimin*phimax < 0. &&           //Handle pi border:
	phimax - phimin > Geom::pi()) { //Assume that the Det is on
      //the shortest side 
      std::swap(phimin,phimax);
    }
    phiEdge.push_back(std::pair<double,double>(phimin,phimax));
  }
  
  for (unsigned int i = 0; i < theNbins; i++) {
    
    // Put the two phi values in the [0,2pi] range
    double firstPhi  = positiveRange(phiEdge[i].first);
    double secondPhi = positiveRange(phiEdge[binIndex(i-1)].second);
    
    // Reformat them in the [-pi,pi] range
    Geom::Phi<double> firstEdge(firstPhi);
    Geom::Phi<double> secondEdge(secondPhi);
    
    // Calculate the mean and format the result in the [-pi,pi] range
    // Get their value in order to perform the mean in the correct way
    Geom::Phi<double> mean((firstEdge.value() + secondEdge.value())/2.);
    
    // Special case: at +-pi there is a discontinuity
    if ( phiEdge[i].first * phiEdge[binIndex(i-1)].second < 0 &&
	 fabs(firstPhi-secondPhi) < Geom::pi() ) 
      mean = Geom::pi() - mean;
    
    thePhiBorders.push_back(mean);
  }
  
  for (unsigned int i = 0; i < theNbins; i++) {
    if (Geom::Phi<double>(phiEdge[i].first)
	- Geom::Phi<double>(phiEdge[binIndex(i-1)].second) < 0) {
      isPhiOverlapping_ = true;
      break;
    }
  }
  
  double rms = stat_RMS(spread); 
  if ( rms < 0.01*step) { 
    isPhiPeriodic_ = true;
  }
  
  //Check that everything is proper
  if (thePhiBorders.size() != theNbins || thePhiBins.size() != theNbins) 
    //FIXME
    throw cms::Exception("UnexpectedState") << "PhiBorderFinder: consistency error";
}
