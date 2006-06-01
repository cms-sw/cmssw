#include "RecoTracker/TkNavigation/interface/LayerCollector.h"
#include "RecoTracker/TkNavigation/interface/StartingLayerFinder.h" 

using namespace std;

vector<const DetLayer*> LayerCollector::allLayers(const FTS& aFts) const {

  std::cout << "  LayerCollector::allLayers " << std::endl;
  vector<const DetLayer*> myLayers;
  vector<const DetLayer*> compatibleLayers;


  FTS myFts(aFts.parameters());
  std::cout << "  LayerCollector::allLayers initial FTS parameters " << aFts.parameters().position() << " " << aFts.parameters().momentum() <<  std::endl;  
  vector<const DetLayer*> nextLayers = finder()->startingLayers(myFts, deltaR(), deltaZ());

  vector<const DetLayer*> dummy;


  std::cout << "  LayerCollector::allLayers nextLayers.size() frpm the layer finder " << nextLayers.size()  << std::endl;

  bool inside = true;
  while(inside) {

    inside = false;
    for(vector<const DetLayer*>::iterator ilay = nextLayers.begin(); ilay != nextLayers.end(); ilay++) {

      
      TSOS pTsos = propagator()->propagate(myFts, (**ilay).surface());
      std::cout << "  LayerCollector::allLayers after propagation TSOS parameters " << pTsos.globalParameters().position() << " " << pTsos.globalParameters().momentum() << std::endl;
 


      if(pTsos.isValid()) {

	inside = true;

     

	if((**ilay).part() == barrel) {
	  std::cout << "  LayerCollector::allLayers part Barrel " <<  std::endl;
	  Range barrZRange((**ilay).position().z() - 
			   0.5*((**ilay).surface().bounds().length()),
			   (**ilay).position().z() + 
			   0.5*((**ilay).surface().bounds().length()));
	  Range trajZRange(pTsos.globalPosition().z() - deltaZ(),
			   pTsos.globalPosition().z() + deltaZ());
	  
	  if(rangesIntersect(trajZRange, barrZRange)) 
	    myLayers.push_back(*ilay);

	} else if((**ilay).part() == forward) {
	  std::cout << "  LayerCollector::allLayers part Forwd " << std::endl;
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

	compatibleLayers = (**ilay).compatibleLayers(*pTsos.freeState(), 
					 propagator()->propagationDirection());
	break;


      }     



    }
  }

  std::cout << "  nextLayers.size() accessible from this one " << nextLayers.size()  << std::endl;
  std::cout << "  compatibleLayers.size() compatible with this one " << compatibleLayers.size()  << std::endl;
  
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








