#include "RecoTracker/TkNavigation/interface/LayerCollector.h"
#include "RecoTracker/TkNavigation/interface/StartingLayerFinder.h" 

using namespace std;

vector<const DetLayer*> LayerCollector::allLayers(const FTS& aFts) const {


  vector<const DetLayer*> myLayers;



  FTS myFts(aFts.parameters());

  vector<const DetLayer*> nextLayers = finder()->startingLayers(myFts, deltaR(), deltaZ());

  vector<const DetLayer*> dummy;




  bool inside = true;
  while(inside) {

    inside = false;
    for(vector<const DetLayer*>::iterator ilay = nextLayers.begin(); ilay != nextLayers.end(); ilay++) {

      
      TSOS pTsos = propagator()->propagate(myFts, (**ilay).surface());


      if(pTsos.isValid()) {

	inside = true;

     

	if((**ilay).location() == GeomDetEnumerators::barrel) {

	  Range barrZRange((**ilay).position().z() - 
			   0.5*((**ilay).surface().bounds().length()),
			   (**ilay).position().z() + 
			   0.5*((**ilay).surface().bounds().length()));
	  Range trajZRange(pTsos.globalPosition().z() - deltaZ(),
			   pTsos.globalPosition().z() + deltaZ());
	  
	  if(rangesIntersect(trajZRange, barrZRange)) 
	    myLayers.push_back(*ilay);

	} else if((**ilay).location() == GeomDetEnumerators::endcap) {

	  const ForwardDetLayer* fwd = 
	    dynamic_cast<const ForwardDetLayer*>(*ilay);
	  Range fwdRRange((*fwd).specificSurface().innerRadius(),
			  (*fwd).specificSurface().outerRadius());
	  Range trajRRange(pTsos.globalPosition().perp() - deltaR(),
			   pTsos.globalPosition().perp() + deltaR());

	  if(rangesIntersect(trajRRange, fwdRRange)) 
	    myLayers.push_back(*ilay);
	 
	}

	myFts = FTS(pTsos.globalParameters());
 
	
	nextLayers = (**ilay).nextLayers(*pTsos.freeState(), 
					 propagator()->propagationDirection());


	break;


      }     



    }
  }



  
  return myLayers;
}

vector<const BarrelDetLayer*> LayerCollector::barrelLayers(const FTS& aFts) const {

  vector<const DetLayer*> all = allLayers(aFts);
  vector<const BarrelDetLayer*> barrelLayers;


  for(vector<const DetLayer*>::iterator ilay = all.begin();
      ilay != all.end(); ilay++) {

    if(const BarrelDetLayer* myBarrel = 
       dynamic_cast<const BarrelDetLayer*>(*ilay))
      barrelLayers.push_back(myBarrel);
  }


  return barrelLayers;
}
  
vector<const ForwardDetLayer*> LayerCollector::forwardLayers(const FTS& aFts) const {
  
  vector<const DetLayer*> all = allLayers(aFts);
  vector<const ForwardDetLayer*> fwdLayers;


  for(vector<const DetLayer*>::iterator ilay = all.begin();
      ilay != all.end(); ilay++) {
    
    if(const ForwardDetLayer* myFwd = 
       dynamic_cast<const ForwardDetLayer*>(*ilay))
      fwdLayers.push_back(myFwd);
  }

  
  return fwdLayers;
}








