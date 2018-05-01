#include "TrackingTools/DetLayers/interface/RBorderFinder.h"

RBorderFinder::RBorderFinder(const std::vector<const Det*>& utheDets) 
  : theNbins(utheDets.size()), 
    isRPeriodic_(false), 
    isROverlapping_(false)
{
  std::vector<const Det*> theDets = utheDets;
  precomputed_value_sort(theDets.begin(), theDets.end(), DetR());
  
  std::vector<ConstReferenceCountingPointer<BoundDisk> > disks(theNbins);
  for ( int i = 0; i < theNbins; i++ ) {
    disks[i] = 
      dynamic_cast<const BoundDisk*> (&(theDets[i]->surface()));
    if (disks[i]==nullptr) {
      throw cms::Exception("UnexpectedState") << "RBorderFinder: implemented for BoundDisks only";
    }
  }
  
  
  if (theNbins==1) { // Trivial case
    isRPeriodic_ = true; // meaningless in this case
    theRBorders.push_back(disks.front()->innerRadius());
    theRBins.push_back((disks.front()->outerRadius()+disks.front()->innerRadius()));
    //       std::cout << "RBorderFinder:  theNbins " << theNbins << std::endl
    // 		<< " C: " << theRBins[0]
    // 		<< " Border: " << theRBorders[0] << std::endl;
  } else { // More than 1 bin
    double step = (disks.back()->innerRadius() -
		   disks.front()->innerRadius())/(theNbins-1);
    std::vector<double> spread;
    std::vector<std::pair<double,double> > REdge;
    REdge.reserve(theNbins);
    theRBorders.reserve(theNbins);
    theRBins.reserve(theNbins);
    spread.reserve(theNbins);
    
    for ( int i = 0; i < theNbins; i++ ) {
      theRBins.push_back((disks[i]->outerRadius()+disks[i]->innerRadius())/2.);
      spread.push_back(theRBins.back() - (theRBins[0] + i*step));
      REdge.push_back(std::pair<double,double>(disks[i]->innerRadius(),
					       disks[i]->outerRadius()));
    }
    
    theRBorders.push_back(REdge[0].first);
    for (int i = 1; i < theNbins; i++) {
      // Average borders of previous and next bins
      double br = (REdge[(i-1)].second + REdge[i].first)/2.;
      theRBorders.push_back(br);
    }
    
    for (int i = 1; i < theNbins; i++) {
      if (REdge[i].first - REdge[i-1].second < 0) {
	isROverlapping_ = true;
	break;
      }
    }
    
    double rms = stat_RMS(spread); 
    if ( rms < 0.01*step) { 
      isRPeriodic_ = true;
    }
  }
  
  //Check that everything is proper
  if ((int)theRBorders.size() != theNbins || (int)theRBins.size() != theNbins) 
    throw cms::Exception("UnexpectedState") << "RBorderFinder consistency error";
}
