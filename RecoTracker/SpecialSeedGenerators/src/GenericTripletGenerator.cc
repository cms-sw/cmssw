#include "RecoTracker/SpecialSeedGenerators/interface/GenericTripletGenerator.h"
#include "RecoTracker/TkSeedGenerator/interface/FastCircle.h"
typedef SeedingHitSet::ConstRecHitPointer SeedingHit;


#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <map>
using namespace ctfseeding;


GenericTripletGenerator::GenericTripletGenerator(const edm::ParameterSet& conf, edm::ConsumesCollector& iC):
  theSeedingLayerToken(iC.consumes<SeedingLayerSetsHits>(conf.getParameter<edm::InputTag>("LayerSrc"))) {
	edm::LogInfo("CtfSpecialSeedGenerator|GenericTripletGenerator") << "Constructing GenericTripletGenerator";
} 


const OrderedSeedingHits& GenericTripletGenerator::run(const TrackingRegion& region,
                              				     const edm::Event& e,
                              				     const edm::EventSetup& es){
	hitTriplets.clear();
	hitTriplets.reserve(0);
	edm::Handle<SeedingLayerSetsHits> hlayers;
	e.getByToken(theSeedingLayerToken, hlayers);
	const SeedingLayerSetsHits& layers = *hlayers;
        if(layers.numberOfLayersInSet() != 3)
          throw cms::Exception("CtfSpecialSeedGenerator") << "You are using " << layers.numberOfLayersInSet() <<" layers in set instead of 3 ";
	std::map<float, OrderedHitTriplet> radius_triplet_map;
        for(SeedingLayerSetsHits::SeedingLayerSet ls: layers) {
		auto innerHits  = region.hits(e, es, ls[0]);
		//std::cout << "innerHits.size()=" << innerHits.size() << std::endl;
		auto middleHits = region.hits(e, es, ls[1]);
		//std::cout << "middleHits.size()=" << middleHits.size() << std::endl;
		auto outerHits  = region.hits(e, es, ls[2]);
		//std::cout << "outerHits.size()=" << outerHits.size() << std::endl;
		//std::cout << "trying " << innerHits.size()*middleHits.size()*outerHits.size() << " combinations "<<std::endl;
		for (auto iOuterHit = outerHits.begin(); iOuterHit != outerHits.end(); iOuterHit++){
		  for (auto iMiddleHit = middleHits.begin(); iMiddleHit != middleHits.end(); iMiddleHit++){
		    for (auto iInnerHit = innerHits.begin(); iInnerHit != innerHits.end(); iInnerHit++){
		      //GlobalPoint innerpos  = ls[0].hitBuilder()->build(&(**iInnerHit))->globalPosition();
		      //GlobalPoint middlepos = ls[1].hitBuilder()->build(&(**iMiddleHit))->globalPosition();
		      //GlobalPoint outerpos  = ls[2].hitBuilder()->build(&(**iOuterHit))->globalPosition();
		      //FastCircle circle(innerpos,
		      //  		  middlepos,
		      //                  outerpos);
		      //do a first check on the radius of curvature to reduce the combinatorics	
		      OrderedHitTriplet oht(&(**iInnerHit),&(**iMiddleHit),&(**iOuterHit));
		      std::pair<bool,float> val_radius = qualityFilter(oht,radius_triplet_map,ls);
		      if (val_radius.first){
			//if (circle.rho() > 200 || circle.rho() == 0) { //0 radius means straight line
			//hitTriplets.push_back(OrderedHitTriplet(*iInnerHit,
			//					*iMiddleHit,
			//					*iOuterHit));
			hitTriplets.push_back(oht);
			radius_triplet_map.insert(std::make_pair(val_radius.second,oht));
		      }
		    }
		  }
		}
        }
	//std::cout << "ending with " << hitTriplets.size() << " triplets" << std::endl;
	return hitTriplets;
}

std::pair<bool,float> GenericTripletGenerator::qualityFilter(const OrderedHitTriplet& oht, 
							     const std::map<float, OrderedHitTriplet>& map,
							     const SeedingLayerSetsHits::SeedingLayerSet& ls) const{
  //first check the radius
  GlobalPoint innerpos  = oht.inner()->globalPosition();
  GlobalPoint middlepos = oht.middle()->globalPosition();
	GlobalPoint outerpos  = oht.outer()->globalPosition();
	std::vector<const TrackingRecHit*> ohttrh;
	ohttrh.push_back(&(*(oht.inner()))); ohttrh.push_back(&(*(oht.middle()))); ohttrh.push_back(&(*(oht.outer()))); 
	std::vector<const TrackingRecHit*>::const_iterator ioht;
	//now chech that the change in phi is reasonable. the limiting angular distance is the one in case 
	//one of the two points is a tangent point.
	float limit_phi_distance1 = sqrt((middlepos.x()-outerpos.x())*(middlepos.x()-outerpos.x()) + 
					 (middlepos.y()-outerpos.y())*(middlepos.y()-outerpos.y()))/middlepos.mag();//actually this is the tangent of the limiting angle		 
	float limit_phi_distance2 = sqrt((middlepos.x()-innerpos.x())*(middlepos.x()-innerpos.x()) +
					 (middlepos.y()-innerpos.y())*(middlepos.y()-innerpos.y()))/innerpos.mag();
	//if (fabs(tan(outerpos.phi()-middlepos.phi()))>limit_phi_distance1 || 
	//    fabs(tan(innerpos.phi()-middlepos.phi()))>limit_phi_distance2) {
	if (fabs(outerpos.phi()-middlepos.phi())>fabs(atan(limit_phi_distance1)) ||
            fabs(innerpos.phi()-middlepos.phi())>fabs(atan(limit_phi_distance2)) ) {	
	  //std::cout << "rejected because phi" << std::endl;
	  return std::make_pair(false, 0.);
	}
	//should we add a control on the r-z plane too?
	/*
 	//now check that there is no big change in the r-z projection
	float dz1 = outerpos.z()-middlepos.z();
	float dr1 = sqrt(outerpos.x()*outerpos.x()+outerpos.y()*outerpos.y())-
	sqrt(middlepos.x()*middlepos.x()+middlepos.y()*middlepos.y()); 		
	float dz2 = middlepos.z()-innerpos.z();	
	float dr2 = sqrt(middlepos.x()*middlepos.x()+middlepos.y()*middlepos.y())-
	sqrt(innerpos.x()*innerpos.x()+innerpos.y()*innerpos.y());
	float tan1 = dz1/dr1;
	float tan2 = dz2/dr2;
	//how much should we allow? should we make it configurable?
	if (fabs(tan1-tan2)/tan1>0.5){
	//std::cout << "rejected because z" << std::endl;
	return std::make_pair(false, 0.);	
	}
	*/
	//now check the radius is not too small
	FastCircle circle(innerpos, middlepos, outerpos);
	if (circle.rho() < 200 && circle.rho() != 0) return std::make_pair(false, circle.rho()); //to small radius 	
	//now check if at least 2 hits are shared with an existing triplet
	//look for similar radii in the map
	std::map<float, OrderedHitTriplet>::const_iterator lower_bound = map.lower_bound((1-0.01)*circle.rho());
	std::map<float, OrderedHitTriplet>::const_iterator upper_bound = map.upper_bound((1+0.01)*circle.rho());	
	std::map<float, OrderedHitTriplet>::const_iterator iter;
	for (iter = lower_bound; iter != upper_bound && iter->first <= upper_bound->first; iter++){
	  int shared=0;
	  std::vector<const TrackingRecHit*> curtrh;
	  curtrh.push_back(&*(iter->second.inner()));curtrh.push_back(&*(iter->second.middle()));curtrh.push_back(&*(iter->second.outer()));
	  std::vector<const TrackingRecHit*>::const_iterator curiter;
	  for (curiter = curtrh.begin(); curiter != curtrh.end(); curiter++){
	    for (ioht = ohttrh.begin(); ioht != ohttrh.end(); ioht++){
	      if ((*ioht)->geographicalId()==(*curiter)->geographicalId() && 
		  ((*ioht)->localPosition()-(*curiter)->localPosition()).mag()<1e-5) shared++;	
	    }
	  }
	  if (shared>1) return std::make_pair(false, circle.rho());	
	}	
	
	return std::make_pair(true,circle.rho());
}
