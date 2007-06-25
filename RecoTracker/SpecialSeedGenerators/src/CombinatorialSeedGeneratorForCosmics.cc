#include "RecoTracker/SpecialSeedGenerators/interface/CombinatorialSeedGeneratorForCosmics.h"
#include "RecoTracker/TkHitPairs/interface/CosmicLayerPairs.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/CommonDetAlgo/interface/GlobalError.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/GluedGeomDet.h"
#include "Geometry/Surface/interface/BoundCylinder.h"
#include "Geometry/Surface/interface/RectangularPlaneBounds.h"
#include "Geometry/Surface/interface/BoundPlane.h"
#include "Geometry/Surface/interface/GloballyPositioned.h"
#include "Geometry/CommonDetAlgo/interface/ErrorFrameTransformer.h"
#include "TrackingTools/GeomPropagators/interface/StraightLinePlaneCrossing.h"


CombinatorialSeedGeneratorForCosmics::CombinatorialSeedGeneratorForCosmics(edm::ParameterSet const& conf): //SeedGeneratorFromTrackingRegion(conf),
  conf_(conf)
{
  edm::LogVerbatim("CombinatorialSeedGeneratorForCosmics") << "Constructing CombinatorialSeedGeneratorForCosmics";
  geometry=conf_.getUntrackedParameter<std::string>("GeometricStructure","STANDARD");
  p = conf_.getParameter<double>("SeedMomentum");
  useScintillatorsConstraint = conf_.getParameter<bool>("UseScintillatorsConstraint");
  if (useScintillatorsConstraint){	
	  edm::ParameterSet upperScintPar = conf_.getParameter<edm::ParameterSet>("UpperScintillatorParameters");
	  edm::ParameterSet lowerScintPar = conf_.getParameter<edm::ParameterSet>("LowerScintillatorParameters");
	  //cout << "WidthInX" << upperScintPar.getParameter<double>("WidthInX") << endl;
	  //creates rectangular bounds for upper scintillator, arguments are half size 
	  RectangularPlaneBounds upperBounds(upperScintPar.getParameter<double>("WidthInX"),
					     upperScintPar.getParameter<double>("LenghtInZ"),
					     1);
	  //cout << "GlobalX" << upperScintPar.getParameter<double>("GlobalX") << endl;
	  //places the upper scintillator according to position in cfg
	  GlobalPoint upperPosition(upperScintPar.getParameter<double>("GlobalX"),
				     upperScintPar.getParameter<double>("GlobalY"),
			     upperScintPar.getParameter<double>("GlobalZ"));
	  //cout << "Upper position x, y, z " << upperPosition.x() << ", " << upperPosition.y() << ", " << upperPosition.z() << endl;
	  edm::LogVerbatim("CombinatorialSeedGeneratorForCosmics") << "Upper Scintillator position x, y, z " << upperPosition.x() << ", " << upperPosition.y() << ", " << upperPosition.z();
	  //creates rectangular bounds for lower scintillator, arguments are half size
	  RectangularPlaneBounds lowerBounds(lowerScintPar.getParameter<double>("WidthInX"),
					     lowerScintPar.getParameter<double>("LenghtInZ"),
					     1);
	  //cout << "bound lenght " << lowerBounds.length() << endl;
	  //places the lower scintillator according to position in cfg		
	  GlobalPoint lowerPosition(lowerScintPar.getParameter<double>("GlobalX"),
				     lowerScintPar.getParameter<double>("GlobalY"),
				     lowerScintPar.getParameter<double>("GlobalZ"));
  	edm::LogVerbatim("CombinatorialSeedGeneratorForCosmics") << "Lower Scintillator position x, y, z " << lowerPosition.x() << ", " << lowerPosition.y() << ", " << lowerPosition.z() ;
  	TkRotation<float> rot(1,0,0,0,0,1,0,1,0);
  	//cout << "matrix " << rot.xx() << rot.xy() << rot.xz() << rot.yx() << rot.yy() << rot.yz() << rot.zx() << rot.zy() << rot.zz()<< endl; 	
  	upperScintillator = new BoundPlane(upperPosition, rot, &upperBounds);
  	//cout << "upperPosition" << upperScintillator->toGlobal(LocalPoint(0,0,0)) << endl;	
  	lowerScintillator = new BoundPlane(lowerPosition, rot, &lowerBounds);	
  	//cout << "lowerPosition" << lowerScintillator->toGlobal(LocalPoint(0,0,0)) << endl;	
  } else {
	upperScintillator = 0;
	lowerScintillator = 0;
  }

  		
  produces<TrajectorySeedCollection>();
}

CombinatorialSeedGeneratorForCosmics::~CombinatorialSeedGeneratorForCosmics(){
	if (upperScintillator) {delete upperScintillator; upperScintillator = 0;}
	if (lowerScintillator) {delete lowerScintillator; lowerScintillator = 0;}
}

void CombinatorialSeedGeneratorForCosmics::produce(edm::Event& e, const edm::EventSetup& es)
{
  // get Inputs
  edm::InputTag matchedrecHitsTag = conf_.getParameter<edm::InputTag>("matchedRecHits");
  edm::InputTag rphirecHitsTag = conf_.getParameter<edm::InputTag>("rphirecHits");
  edm::InputTag stereorecHitsTag = conf_.getParameter<edm::InputTag>("stereorecHits");

  edm::Handle<SiStripRecHit2DCollection> rphirecHits;
  e.getByLabel( rphirecHitsTag, rphirecHits );
  edm::Handle<SiStripRecHit2DCollection> stereorecHits;
  e.getByLabel( stereorecHitsTag ,stereorecHits );
  edm::Handle<SiStripMatchedRecHit2DCollection> matchedrecHits;
  e.getByLabel( matchedrecHitsTag ,matchedrecHits );
  
  std::auto_ptr<TrajectorySeedCollection> output(new TrajectorySeedCollection);

  init(*stereorecHits,*rphirecHits,*matchedrecHits,es);

  run(*output,es);

  edm::LogVerbatim("Algorithm Performance") << " number of seeds = "<< output->size();
  e.put(output);
}

void 
CombinatorialSeedGeneratorForCosmics::init(const SiStripRecHit2DCollection &collstereo,
			      const SiStripRecHit2DCollection &collrphi ,
			      const SiStripMatchedRecHit2DCollection &collmatched,
			      const edm::EventSetup& iSetup)
{
  iSetup.get<IdealMagneticFieldRecord>().get(magfield);
  iSetup.get<TrackerDigiGeometryRecord>().get(tracker);
  //get matcher
  std::string matcher = conf_.getParameter<std::string>("Matcher");
  iSetup.get<TrackerCPERecord>().get(matcher, rechitmatcher);


  stereocollection = &(collstereo);

  edm::LogVerbatim("CombinatorialSeedGeneratorForCosmics") << "Initializing...";
  CosmicLayerPairs cosmiclayers;
  HitPairs.clear();
  meanRadius = 0;
  cosmiclayers.init(collstereo,collrphi,collmatched,geometry,iSetup);
  std::vector<SeedLayerPairs::LayerPair> layerPairs = cosmiclayers();
  std::vector<SeedLayerPairs::LayerPair>::const_iterator iLayerPairs;
  for(iLayerPairs = layerPairs.begin(); iLayerPairs != layerPairs.end(); iLayerPairs++){
	const LayerWithHits* outer = (*iLayerPairs).second;
	const LayerWithHits* inner = (*iLayerPairs).first;
	if ((!inner) || (!outer)) continue;
	meanRadius += ( (BoundCylinder*) &(outer->layer()->surface()) )->radius()+
		      (	(BoundCylinder*) &(inner->layer()->surface()) )->radius();
	std::vector<const TrackingRecHit*>::const_iterator iOuterHit;
	for (iOuterHit = outer->recHits().begin(); iOuterHit != outer->recHits().end(); iOuterHit++){
		std::vector<const TrackingRecHit*>::const_iterator iInnerHit;
		for (iInnerHit = inner->recHits().begin(); iInnerHit != inner->recHits().end(); iInnerHit++){
			if ((*iInnerHit) && (*iOuterHit)){
			  HitPairs.push_back( OrderedHitPair(*iInnerHit, *iOuterHit) );
			}
		}
	}
	
  }
  //FIXME in the following we use meanRadius to decide if the seed from which hit pair the seed is made. dangerous
  if (layerPairs.size()) meanRadius /= (layerPairs.size()*2);
  //std::cout << "mean radius " << meanRadius << std::endl;
}


void CombinatorialSeedGeneratorForCosmics::run(TrajectorySeedCollection &output,const edm::EventSetup& iSetup){
  //run the algorithm
  seeds(output,iSetup);
}
void CombinatorialSeedGeneratorForCosmics::seeds(TrajectorySeedCollection &output,
				    const edm::EventSetup& iSetup){
	for(uint is=0;is<HitPairs.size();is++){
		//inner point position in global coordinates
    		GlobalPoint inner = tracker->idToDet(HitPairs[is].inner()->geographicalId())->surface().toGlobal(HitPairs[is].inner()->localPosition());
		//outer point position in global coordinates 
    		GlobalPoint outer = tracker->idToDet(HitPairs[is].outer()->geographicalId())->surface().toGlobal(HitPairs[is].outer()->localPosition());
    		edm::OwnVector<TrackingRecHit> hits;
		//for the moment the seed is only built from hits at positive y
		//the inner hit must have a y less than the outer hit		
	      	if((outer.y()>0)&&(inner.y()>0)&&((outer.y()-inner.y())>0)){
			GlobalVector momentum = GlobalVector(inner-outer).unit();
			//momentum*=p;
			//decide propagation direction according to innermost or outermost pair
			PropagationDirection dir;
                        GlobalPoint* firstPoint;
                        const TrackingRecHit* firstHit;
			const TrackingRecHit* secondHit;
			//if the inner hit is at a radius < meanRadius we are building a seed from inner layers
			//prop direction and hit order enstablished accordingly
                        if ( (inner.perp() < meanRadius) ){
                                dir = oppositeToMomentum;
                                firstPoint = &inner;
                                firstHit = HitPairs[is].inner();
				secondHit = HitPairs[is].outer();
                        } else if ( (outer.perp() > meanRadius)) {  // otherwise we are building a seed from outer layers
                                dir = alongMomentum;
                                firstPoint = &outer;
                                firstHit = HitPairs[is].outer();
				secondHit = HitPairs[is].inner();
                        } else {std::cout << "unable to determine direction outer " << outer.perp()
						<< " inner " << inner.perp() << std::endl; return;}			
			//try to match both hits
			//the direction is the one obtained from rphi hit only	
			edm::OwnVector<TrackingRecHit> firstMatch = match(firstHit, momentum); 
			edm::OwnVector<TrackingRecHit> secondMatch = match(secondHit, momentum);
			//now build seeds
			edm::OwnVector<TrackingRecHit>::const_iterator iFirst;
			edm::OwnVector<TrackingRecHit>::const_iterator iSecond;
			for (iFirst = firstMatch.begin(); iFirst != firstMatch.end(); iFirst++){
				for (iSecond = secondMatch.begin(); iSecond != secondMatch.end(); iSecond++){
					//edm::OwnVector<TrajectorySeed> tmp = buildSeed(&(*iFirst), &(*iSecond), dir);
					const TrajectorySeed* tmp = buildSeed(&(*iFirst), &(*iSecond), dir);
					if (tmp){
						//cout << "SEED ----> " << iTmp->startingState().parameters().position() << endl;
						//output.push_back(*(iTmp->clone()));
						output.push_back(*tmp);
						delete tmp;
					}
				}
			}
	      
      		}
  	}
  //std::cout << "End of CombinatorialSeedGeneratorForCosmics::seeds" << std::endl;
}

bool CombinatorialSeedGeneratorForCosmics::checkDirection(const FreeTrajectoryState& state, 
							  const MagneticField* magField){
	if (!useScintillatorsConstraint) return true;
	//StraightLinePropagator prop(magField);
	StraightLinePlaneCrossing planeCrossingLower( Basic3DVector<float>(state.position()), 
						 Basic3DVector<float>(state.momentum()),
						 alongMomentum);
	StraightLinePlaneCrossing planeCrossingUpper( Basic3DVector<float>(state.position()),
                                                 Basic3DVector<float>(state.momentum()),
                                                 oppositeToMomentum);
	//std::pair<bool,double> pathLengthUpper =  planeCrossingUpper.pathLength(*upperScintillator);
	//std::pair<bool,double> pathLengthLower =  planeCrossingLower.pathLength(*lowerScintillator);
	std::pair<bool,StraightLinePlaneCrossing::PositionType> positionUpper = planeCrossingUpper.position(*upperScintillator);
	std::pair<bool,StraightLinePlaneCrossing::PositionType> positionLower = planeCrossingLower.position(*lowerScintillator);
	//TSOS upperTSOS = prop.propagate(state, *upperScintillator); 	
	//TSOS lowerTSOS = prop.propagate(state, *lowerScintillator); 	
	if (!(positionUpper.first && positionLower.first)) {
		edm::LogInfo("CombinatorialSeedGeneratorForCosmics") << "Scintillator plane not crossed";
		return false;
	}
	LocalPoint positionUpperLocal = upperScintillator->toLocal((GlobalPoint)(positionUpper.second));
        LocalPoint positionLowerLocal = lowerScintillator->toLocal((GlobalPoint)(positionLower.second));
	if (upperScintillator->bounds().inside(positionUpperLocal) && lowerScintillator->bounds().inside(positionLowerLocal)) {
		edm::LogInfo("CombinatorialSeedGeneratorForCosmics") << "position on Upper scintillator " << positionUpper.second;
		edm::LogInfo("CombinatorialSeedGeneratorForCosmics") << "position on Lower scintillator " << positionLower.second;
		
		return true;
	}
	edm::LogInfo("CombinatorialSeedGeneratorForCosmics") << "scintillator not crossed in bounds: position on Upper scintillator " << positionUpper.second << " position on Lower scintillator " << positionLower.second;
	//cout << "SKIPPING" << endl;
	return false;
}

edm::OwnVector<TrackingRecHit> CombinatorialSeedGeneratorForCosmics::match(const TrackingRecHit* hit, 
						const GlobalVector& direction){
	edm::OwnVector<TrackingRecHit> hits;
	const std::vector<DetId> stereodetIDs = stereocollection->ids();
	StripSubdetector monoDetId(hit->geographicalId());//assumes that the hit pair contains mono hit
	unsigned int stereoId = 0;
	stereoId=monoDetId.partnerDetId();
	DetId stereoDetId(stereoId);
	std::vector<DetId>::const_iterator partnerdetiter=
		std::find(stereodetIDs.begin(),stereodetIDs.end(),stereoDetId);
		if(partnerdetiter==stereodetIDs.end()) stereoId=0;

	const SiStripRecHit2DCollection::range rhpartnerRange = stereocollection->get(stereoDetId);
	SiStripRecHit2DCollection::const_iterator rhpartnerRangeIteratorBegin = rhpartnerRange.first;
	SiStripRecHit2DCollection::const_iterator rhpartnerRangeIteratorEnd   = rhpartnerRange.second;
	edm::OwnVector<SiStripMatchedRecHit2D> collectorMatchedSingleHit;

	if (stereoId>0){  //if it exists a stereo det with hits
		const GluedGeomDet* gluedDet = (const GluedGeomDet*) tracker->idToDet(DetId(monoDetId.glued()));
		//GeomDetUnit * detUnit = tracker->idToDetUnit(DetId(monoDetId.glued()));
		collectorMatchedSingleHit=rechitmatcher->match((const SiStripRecHit2D*)hit,rhpartnerRangeIteratorBegin,rhpartnerRangeIteratorEnd,gluedDet,(gluedDet->surface()).toLocal(direction));
		if (collectorMatchedSingleHit.size() > 0){
			edm::OwnVector<SiStripMatchedRecHit2D>::const_iterator iter;
			for(iter = collectorMatchedSingleHit.begin(); iter != collectorMatchedSingleHit.end(); iter++){
				hits.push_back(new SiStripMatchedRecHit2D(*iter));
			}
                } else hits.push_back(hit->clone());
        } else hits.push_back(hit->clone());
	return hits;
}

const TrajectorySeed* CombinatorialSeedGeneratorForCosmics::buildSeed(const TrackingRecHit* first, 
					             const TrackingRecHit* second,
						     const PropagationDirection& dir){
	//edm::OwnVector<TrajectorySeed> outseed;
	//calculates position and error for the two hits in global frame
	std::pair<GlobalPoint, GlobalError> firstHitPosition = toGlobal(first);
	//cout << "First hit position " << firstHitPosition.first << " with error " << firstHitPosition.second.matrix() << endl; 
	std::pair<GlobalPoint, GlobalError> secondHitPosition = toGlobal(second);
	//cout << "Second hit position " << secondHitPosition.first << " with error " << secondHitPosition.second.matrix() << endl; 


	GlobalVector point_difference;
	//FIXME the momentum direction is recovered from propagation direction. potentially dangerous
	if (dir == alongMomentum) point_difference = GlobalVector(secondHitPosition.first-firstHitPosition.first);
	else  point_difference = GlobalVector(firstHitPosition.first-secondHitPosition.first);


	GlobalVector momentum = point_difference.unit()*p;
	//float modulus = point_difference.mag();
		
	
	GlobalTrajectoryParameters Gtp(firstHitPosition.first,
				       momentum,
				       -1,
				       &(*magfield));



	AlgebraicSymMatrix startingCurvilinearError(5,1);
	startingCurvilinearError[0][0]=1/p;
	startingCurvilinearError[3][3]=firstHitPosition.second.cxx();
	startingCurvilinearError[4][4]=firstHitPosition.second.czz();
	FreeTrajectoryState CosmicSeed(Gtp,CurvilinearTrajectoryError(startingCurvilinearError));

	//check if direction is compatible with scintillators position
	if (!checkDirection(CosmicSeed, &(*magfield))) {return 0;}//outseed;}
	
	//dirty: the information about the kind of the hit if already available in match method
	//do we have projected hit in the seeds?
	//const ProjectedSiStripRecHit2D* phit=dynamic_cast<const ProjectedSiStripRecHit2D*>(first);
        const SiStripMatchedRecHit2D* matchedhit=dynamic_cast<const SiStripMatchedRecHit2D*>(first);
        const SiStripRecHit2D* hit=dynamic_cast<const SiStripRecHit2D*>(first);
        //if(phit) hit=&(phit->originalHit());
	const BoundPlane* plane = 0;
	if (matchedhit){
		const GluedGeomDet * stripdet=(const GluedGeomDet*)tracker->idToDet(matchedhit->geographicalId());
		plane = &(stripdet->surface());
	} else if (hit){
		const StripGeomDetUnit * stripdet=(const StripGeomDetUnit*)tracker->idToDetUnit(hit->geographicalId());
		plane = &(stripdet->surface());
	}
	if (!plane) {return 0;}//outseed;}

	TSOS seedTSOS(CosmicSeed, *plane);
	//rescale error for multiple scattering. how much?
	//seedTSOS.rescaleError(5.0);
	//cout << "Starting TSOS CTF " << seedTSOS << " with error " << seedTSOS.curvilinearError().matrix() <<  endl;
        PTrajectoryStateOnDet *PTraj=
        	transformer.persistentState(seedTSOS, first->geographicalId().rawId());
        edm::OwnVector<TrackingRecHit> seed_hits;
        seed_hits.push_back(first->clone());
	//outseed.push_back(new TrajectorySeed(*PTraj,seed_hits,dir));
	//cout << "SEED -----> "  << trSeed->startingState().parameters().position() << endl;
	//return outseed;
	TrajectorySeed* theSeed = new TrajectorySeed(*PTraj,seed_hits,dir);
	if(PTraj) delete PTraj;
	//return new TrajectorySeed(*PTraj,seed_hits,dir);
	return theSeed;
} 

std::pair<GlobalPoint, GlobalError> CombinatorialSeedGeneratorForCosmics::toGlobal(const TrackingRecHit* rechit){
        GlobalPoint hipos;
        GlobalError error;
        //const ProjectedSiStripRecHit2D* phit=dynamic_cast<const ProjectedSiStripRecHit2D*>(rechit);
        const SiStripMatchedRecHit2D* matchedhit=dynamic_cast<const SiStripMatchedRecHit2D*>(rechit);
        const SiStripRecHit2D* hit=dynamic_cast<const SiStripRecHit2D*>(rechit);
        //if(phit) hit=&(phit->originalHit());
        if(matchedhit){
          LocalPoint position=matchedhit->localPosition();
          DetId id=matchedhit->geographicalId();
          //char* layer = getLayer(id);
          const GluedGeomDet * stripdet=(const GluedGeomDet*)tracker->idToDet(id);
          hipos=stripdet->surface().toGlobal(position);
	  error = ErrorFrameTransformer().transform(rechit->localPositionError(),
                                                    stripdet->surface());
	  //cout << "MATCHED hit position " << hipos << " with error " << error.matrix() << endl; 
          //printPosition(hipos, layer);
	  //error*=5.;	
        }
        else if(hit){
          LocalPoint position=hit->localPosition();
          DetId id=hit->geographicalId();
          //char* layer = getLayer(id);
          const StripGeomDetUnit * stripdet=(const StripGeomDetUnit*)tracker->idToDetUnit(id);
          hipos=stripdet->surface().toGlobal(position);
	  error = ErrorFrameTransformer().transform(rechit->localPositionError(),
                                                    stripdet->surface());
	  //cout << "MONO hit position " << hipos << " with error " << error.matrix() << endl; 
          //printPosition(hipos, layer);
	  //error*=5.;
        }

        return std::make_pair(hipos, error);
}
