#include "RecoTracker/SpecialSeedGenerators/interface/BeamHaloPairGenerator.h"
typedef TransientTrackingRecHit::ConstRecHitPointer SeedingHit;

#include "FWCore/MessageLogger/interface/MessageLogger.h"
using namespace ctfseeding;


BeamHaloPairGenerator::BeamHaloPairGenerator(const edm::ParameterSet& conf, edm::ConsumesCollector& iC): 
  theLayerBuilder(conf.getParameter<edm::ParameterSet>("LayerPSet"), iC)
{
	edm::LogInfo("CtfSpecialSeedGenerator|BeamHaloPairGenerator") << "Constructing BeamHaloPairGenerator";
	theMaxTheta=conf.getParameter<double>("maxTheta");
	theMaxTheta=fabs(sin(theMaxTheta));
} 


const OrderedSeedingHits& BeamHaloPairGenerator::run(const TrackingRegion& region,
                          			    const edm::Event& e,
                              			    const edm::EventSetup& es){
	hitPairs.clear();
        if(theLayerBuilder.check(es)) {
          theLss = theLayerBuilder.layers(es);
        }
	SeedingLayerSets::const_iterator iLss;
	for (iLss = theLss.begin(); iLss != theLss.end(); iLss++){
		SeedingLayers ls = *iLss;
		if (ls.size() != 2){
                	throw cms::Exception("CtfSpecialSeedGenerator") << "You are using " << ls.size() <<" layers in set instead of 2 ";
        	}	
		std::vector<SeedingHit> innerHits  = region.hits(e, es, &ls[0]);
		std::vector<SeedingHit> outerHits  = region.hits(e, es, &ls[1]);
		std::vector<SeedingHit>::const_iterator iOuterHit;
		for (iOuterHit = outerHits.begin(); iOuterHit != outerHits.end(); iOuterHit++){
			std::vector<SeedingHit>::const_iterator iInnerHit;
			for (iInnerHit = innerHits.begin(); iInnerHit != innerHits.end(); iInnerHit++){
			  //do something in there... if necessary
			  const TransientTrackingRecHit::ConstRecHitPointer & crhpi = *iInnerHit;
			  const TransientTrackingRecHit::ConstRecHitPointer & crhpo =  *iOuterHit;
			  GlobalVector d=crhpo->globalPosition() - crhpi->globalPosition();
			  double ABSsinDtheta = fabs(sin(d.theta()));
			  LogDebug("BeamHaloPairGenerator")<<"position1: "<<crhpo->globalPosition()
							   <<" position2: "<<crhpi->globalPosition()
							   <<" |sin(Dtheta)|: "<< ABSsinDtheta <<((ABSsinDtheta>theMaxTheta)?" skip":" keep");

			  if (ABSsinDtheta>theMaxTheta) {;continue;}

			  hitPairs.push_back(OrderedHitPair(*iInnerHit,
							    *iOuterHit));
			}
		}
        }
	return hitPairs;
}
