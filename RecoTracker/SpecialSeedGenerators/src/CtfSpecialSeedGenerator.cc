#include "RecoTracker/SpecialSeedGenerators/interface/CtfSpecialSeedGenerator.h"
//#include "RecoTracker/SpecialSeedGenerators/interface/CosmicLayerTriplets.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DataFormats/GeometrySurface/interface/RectangularPlaneBounds.h"
#include "TrackingTools/GeomPropagators/interface/StraightLinePlaneCrossing.h"

#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducerFactory.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
#include "RecoTracker/TkTrackingRegions/interface/OrderedHitsGeneratorFactory.h"
#include "RecoTracker/TkSeedingLayers/interface/OrderedSeedingHits.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace ctfseeding;

CtfSpecialSeedGenerator::CtfSpecialSeedGenerator(const edm::ParameterSet& conf): 
  conf_(conf),
  requireBOFF(conf.getParameter<bool>("requireBOFF")),
  theMaxSeeds(conf.getParameter<int32_t>("maxSeeds")),
  check(conf,consumesCollector())

{
  	useScintillatorsConstraint = conf_.getParameter<bool>("UseScintillatorsConstraint");
  	edm::LogVerbatim("CtfSpecialSeedGenerator") << "Constructing CtfSpecialSeedGenerator";
  	produces<TrajectorySeedCollection>();
	theSeedBuilder =0; 
	theRegionProducer =0;

	edm::ParameterSet regfactoryPSet = conf_.getParameter<edm::ParameterSet>("RegionFactoryPSet");
  	std::string regfactoryName = regfactoryPSet.getParameter<std::string>("ComponentName");
  	theRegionProducer = TrackingRegionProducerFactory::get()->create(regfactoryName,regfactoryPSet, consumesCollector());
}

CtfSpecialSeedGenerator::~CtfSpecialSeedGenerator(){
}

void CtfSpecialSeedGenerator::endRun(edm::Run const&, edm::EventSetup const&){
    if (theSeedBuilder)    { delete theSeedBuilder;    theSeedBuilder = 0; }
    if (theRegionProducer) { delete theRegionProducer; theRegionProducer = 0; }
    std::vector<OrderedHitsGenerator*>::iterator iGen;	
    for (iGen = theGenerators.begin(); iGen != theGenerators.end(); iGen++){
        delete (*iGen);
    }
    theGenerators.clear();
}

void CtfSpecialSeedGenerator::beginRun(edm::Run const&, const edm::EventSetup& iSetup){
	std::string builderName = conf_.getParameter<std::string>("TTRHBuilder");
        iSetup.get<TransientRecHitRecord>().get(builderName,theBuilder);

        iSetup.get<IdealMagneticFieldRecord>().get(theMagfield);
        iSetup.get<TrackerDigiGeometryRecord>().get(theTracker);

        edm::LogVerbatim("CtfSpecialSeedGenerator") << "Initializing...";
	if (useScintillatorsConstraint){
        	edm::ParameterSet upperScintPar = conf_.getParameter<edm::ParameterSet>("UpperScintillatorParameters");
          	edm::ParameterSet lowerScintPar = conf_.getParameter<edm::ParameterSet>("LowerScintillatorParameters");
          	RectangularPlaneBounds upperBounds(upperScintPar.getParameter<double>("WidthInX"),
                                             upperScintPar.getParameter<double>("LenghtInZ"),
                                             1);
          	GlobalPoint upperPosition(upperScintPar.getParameter<double>("GlobalX"),
                                     	upperScintPar.getParameter<double>("GlobalY"),
                             		upperScintPar.getParameter<double>("GlobalZ"));
          	edm::LogVerbatim("CtfSpecialSeedGenerator")
                	<< "Upper Scintillator position x, y, z " << upperPosition.x()
                	<< ", " << upperPosition.y() << ", " << upperPosition.z();
          	RectangularPlaneBounds lowerBounds(lowerScintPar.getParameter<double>("WidthInX"),
                                             lowerScintPar.getParameter<double>("LenghtInZ"),
                                             1);
          	GlobalPoint lowerPosition(lowerScintPar.getParameter<double>("GlobalX"),
                                     lowerScintPar.getParameter<double>("GlobalY"),
                                     lowerScintPar.getParameter<double>("GlobalZ"));
          	edm::LogVerbatim("CtfSpecialSeedGenerator")
                	<< "Lower Scintillator position x, y, z " << lowerPosition.x()
                	<< ", " << lowerPosition.y() << ", " << lowerPosition.z() ;
          	TkRotation<float> rot(1,0,0,0,0,1,0,1,0);
          	upperScintillator = BoundPlane::build(upperPosition, rot, &upperBounds);
          	lowerScintillator = BoundPlane::build(lowerPosition, rot, &lowerBounds);
  	} 
	
	edm::ESHandle<Propagator>  propagatorAlongHandle;
  	iSetup.get<TrackingComponentsRecord>().get("PropagatorWithMaterial",propagatorAlongHandle);
	edm::ESHandle<Propagator>  propagatorOppositeHandle;
        iSetup.get<TrackingComponentsRecord>().get("PropagatorWithMaterialOpposite",propagatorOppositeHandle);

/*  	edm::ParameterSet hitsfactoryOutInPSet = conf_.getParameter<edm::ParameterSet>("OrderedHitsFactoryOutInPSet");
  	std::string hitsfactoryOutInName = hitsfactoryOutInPSet.getParameter<std::string>("ComponentName");
  	hitsGeneratorOutIn = OrderedHitsGeneratorFactory::get()->create( hitsfactoryOutInName, hitsfactoryOutInPSet);
	std::string propagationDirection = hitsfactoryOutInPSet.getUntrackedParameter<std::string>("PropagationDirection", 
											            "alongMomentum");
	if (propagationDirection == "alongMomentum") outInPropagationDirection = alongMomentum;
	else outInPropagationDirection = oppositeToMomentum;
	edm::LogVerbatim("CtfSpecialSeedGenerator") << "hitsGeneratorOutIn done";

	edm::ParameterSet hitsfactoryInOutPSet = conf_.getParameter<edm::ParameterSet>("OrderedHitsFactoryInOutPSet");
        std::string hitsfactoryInOutName = hitsfactoryInOutPSet.getParameter<std::string>("ComponentName");
        hitsGeneratorInOut = OrderedHitsGeneratorFactory::get()->create( hitsfactoryInOutName, hitsfactoryInOutPSet);

	propagationDirection = hitsfactoryInOutPSet.getUntrackedParameter<std::string>("PropagationDirection",
                                                                                        "alongMomentum");
	if (propagationDirection == "alongMomentum") inOutPropagationDirection = alongMomentum;
        else inOutPropagationDirection = oppositeToMomentum;
	edm::LogVerbatim("CtfSpecialSeedGenerator") << "hitsGeneratorInOut done";
	if (!hitsGeneratorOutIn || !hitsGeneratorInOut) 
		throw cms::Exception("CtfSpecialSeedGenerator") << "Only corcrete implementation GenericPairOrTripletGenerator of OrderedHitsGenerator is allowed ";
*/
	std::vector<edm::ParameterSet> pSets = conf_.getParameter<std::vector<edm::ParameterSet> >("OrderedHitsFactoryPSets");
	std::vector<edm::ParameterSet>::const_iterator iPSet;
	for (iPSet = pSets.begin(); iPSet != pSets.end(); iPSet++){
		std::string hitsfactoryName = iPSet->getParameter<std::string>("ComponentName");
        	theGenerators.push_back(OrderedHitsGeneratorFactory::get()->create( hitsfactoryName, *iPSet));
        	std::string propagationDirection = iPSet->getParameter<std::string>("PropagationDirection");
        	if (propagationDirection == "alongMomentum") thePropDirs.push_back(alongMomentum);
        	else thePropDirs.push_back(oppositeToMomentum);
		std::string navigationDirection = iPSet->getParameter<std::string>("NavigationDirection");              
		if (navigationDirection == "insideOut") theNavDirs.push_back(insideOut);
                else theNavDirs.push_back(outsideIn);
        	edm::LogVerbatim("CtfSpecialSeedGenerator") << "hitsGenerator done";
	} 
	bool setMomentum = conf_.getParameter<bool>("SetMomentum");
	std::vector<int> charges;
	if (setMomentum){
	 	charges = conf_.getParameter<std::vector<int> >("Charges");
	}
	theSeedBuilder = new SeedFromGenericPairOrTriplet(theMagfield.product(), 
							  theTracker.product(), 
							  theBuilder.product(),
							  propagatorAlongHandle.product(),
							  propagatorOppositeHandle.product(),
							  charges,
							  setMomentum,
						          conf_.getParameter<double>("ErrorRescaling"));
	double p = 1;
        if (setMomentum) {
                p = conf_.getParameter<double>("SeedMomentum");
		theSeedBuilder->setMomentumTo(p);
        }

}

void CtfSpecialSeedGenerator::produce(edm::Event& e, const edm::EventSetup& iSetup)
{
  // get Inputs
  std::auto_ptr<TrajectorySeedCollection> output(new TrajectorySeedCollection);
  
  //check on the number of clusters
  if ( !requireBOFF || (theMagfield->inTesla(GlobalPoint(0,0,0)).mag() == 0.00) ) {
      size_t clustsOrZero = check.tooManyClusters(e);
      if (!clustsOrZero){
          bool ok = run(iSetup, e, *output);
          if (!ok) { ; } // nothing to do
      } else edm::LogError("TooManyClusters") << "Found too many clusters (" << clustsOrZero << "), bailing out.\n";
  }
  
  
  edm::LogVerbatim("CtfSpecialSeedGenerator") << " number of seeds = "<< output->size();
  e.put(output);
}

bool CtfSpecialSeedGenerator::run(const edm::EventSetup& iSetup,
					       const edm::Event& e,
					       TrajectorySeedCollection& output){
	std::vector<TrackingRegion*> regions = theRegionProducer->regions(e, iSetup);
	std::vector<TrackingRegion*>::const_iterator iReg;
        bool ok = true;
	for (iReg = regions.begin(); iReg != regions.end(); iReg++){
		if(!theSeedBuilder->momentumFromPSet()) theSeedBuilder->setMomentumTo((*iReg)->ptMin());
		std::vector<OrderedHitsGenerator*>::const_iterator iGen;
		int i = 0;
		for (iGen = theGenerators.begin(); iGen != theGenerators.end(); iGen++){ 
		  ok = buildSeeds(iSetup, 
			     e, 
			     (*iGen)->run(**iReg, e, iSetup),
			     theNavDirs[i], 
			     thePropDirs[i], 
			     output);
		  i++;
                  if (!ok) break;
		}
                if (!ok) break;
	}
	//clear memory
	for (std::vector<TrackingRegion*>::iterator iReg = regions.begin(); iReg != regions.end(); iReg++){
		delete *iReg;
	}
        return ok;
}

bool CtfSpecialSeedGenerator::buildSeeds(const edm::EventSetup& iSetup,
					 const edm::Event& e,
					 const OrderedSeedingHits& osh,
					 const NavigationDirection& navdir,
					 const PropagationDirection& dir,
				         TrajectorySeedCollection& output){ 
  //SeedFromGenericPairOrTriplet seedBuilder(conf_, magfield.product(), tracker.product(), theBuilder.product());
  edm::LogInfo("CtfSpecialSeedGenerator")<<"osh.size() " << osh.size();
  for (unsigned int i = 0; i < osh.size(); i++){
	SeedingHitSet shs = osh[i];
	if (preliminaryCheck(shs,iSetup)){
		std::vector<TrajectorySeed*> seeds = theSeedBuilder->seed(shs, 
							    		dir,
							    		navdir, 
							    		iSetup);
		for (std::vector<TrajectorySeed*>::const_iterator iSeed = seeds.begin(); iSeed != seeds.end(); iSeed++){
		  if (!*iSeed) {edm::LogError("CtfSpecialSeedGenerator")<<"a seed pointer is null. skipping.";continue;}
			if (postCheck(**iSeed)){
				output.push_back(**iSeed);
			}
			delete *iSeed;
			edm::LogVerbatim("CtfSpecialSeedGenerator") << "Seed built";
		}
	}
  }	 
  if ((theMaxSeeds > 0) && (output.size() > size_t(theMaxSeeds))) {
    edm::LogWarning("TooManySeeds") << "Too many seeds ("<< output.size() <<"), bailing out.\n";
    output.clear(); 
    return false;
  }
  return true;
}
//checks the hits are on diffrent layers
bool CtfSpecialSeedGenerator::preliminaryCheck(const SeedingHitSet& shs, const edm::EventSetup &es ){

        edm::ESHandle<TrackerTopology> tTopo;
        es.get<IdealGeometryRecord>().get(tTopo);

	std::vector<std::pair<unsigned int, unsigned int> > vSubdetLayer;
	//std::vector<std::string> vSeedLayerNames;
	bool checkHitsAtPositiveY       = conf_.getParameter<bool>("SeedsFromPositiveY");
	//***top-bottom
	bool checkHitsAtNegativeY       = conf_.getParameter<bool>("SeedsFromNegativeY");
	//***
	bool checkHitsOnDifferentLayers = conf_.getParameter<bool>("CheckHitsAreOnDifferentLayers");
      unsigned int nHits = shs.size();
      for (unsigned int iHit=0; iHit < nHits; ++iHit) {
		//hits for the seeds must be at positive y
            const TrackingRecHit * trh = shs[iHit]->hit();
		TransientTrackingRecHit::RecHitPointer recHit = theBuilder->build(trh);
    		GlobalPoint hitPos = recHit->globalPosition();
		//GlobalPoint point = 
		//  theTracker->idToDet(iHits->geographicalId() )->surface().toGlobal(iHits->localPosition());
		if (checkHitsAtPositiveY){ if (hitPos.y() < 0) return false;}
		//***top-bottom
		if (checkHitsAtNegativeY){ if (hitPos.y() > 0) return false;}
		//***
		//std::string name = iHits->seedinglayer().name(); 
		//hits for the seeds must be in different layers
		unsigned int subid=(*trh).geographicalId().subdetId();
		unsigned int layer = tTopo->layer( (*trh).geographicalId());
		std::vector<std::pair<unsigned int, unsigned int> >::const_iterator iter;
		//std::vector<std::string>::const_iterator iNames;
		if (checkHitsOnDifferentLayers){
			
			for (iter = vSubdetLayer.begin(); iter != vSubdetLayer.end(); iter++){
				if (iter->first == subid && iter->second == layer) return false;
			}
			/*
			for (iNames = vSeedLayerNames.begin(); iNames != vSeedLayerNames.end(); iNames++){
				if (*iNames == name) return false;
			}
			*/
		}
		//vSeedLayerNames.push_back(iHits->seedinglayer().name());
		vSubdetLayer.push_back(std::make_pair(subid, layer));	
	}
	return true;
}


bool CtfSpecialSeedGenerator::postCheck(const TrajectorySeed& seed){
	if (!useScintillatorsConstraint) return true; 
	
        PTrajectoryStateOnDet pstate = seed.startingState();
        TrajectoryStateOnSurface theTSOS = trajectoryStateTransform::transientState(pstate,
								      &(theTracker->idToDet(DetId(pstate.detId()))->surface()),
								      &(*theMagfield));	
	const FreeTrajectoryState* state = theTSOS.freeState();	
	StraightLinePlaneCrossing planeCrossingLower( Basic3DVector<float>(state->position()), 
						      Basic3DVector<float>(state->momentum()),
						      alongMomentum);
        StraightLinePlaneCrossing planeCrossingUpper( Basic3DVector<float>(state->position()),
						      Basic3DVector<float>(state->momentum()),
						      oppositeToMomentum);
        std::pair<bool,StraightLinePlaneCrossing::PositionType> positionUpper = 
	  planeCrossingUpper.position(*upperScintillator);
        std::pair<bool,StraightLinePlaneCrossing::PositionType> positionLower = 
	  planeCrossingLower.position(*lowerScintillator);
        if (!(positionUpper.first && positionLower.first)) {
                 edm::LogVerbatim("CtfSpecialSeedGenerator::checkDirection") 
			<< "Scintillator plane not crossed";
                 return false;
        }
        LocalPoint positionUpperLocal = upperScintillator->toLocal((GlobalPoint)(positionUpper.second));
        LocalPoint positionLowerLocal = lowerScintillator->toLocal((GlobalPoint)(positionLower.second));
        if (upperScintillator->bounds().inside(positionUpperLocal) && 
	    lowerScintillator->bounds().inside(positionLowerLocal)) {
                 edm::LogVerbatim("CtfSpecialSeedGenerator::checkDirection") 
				<< "position on Upper scintillator " 
				<< positionUpper.second;
                 edm::LogVerbatim("CtfSpecialSeedGenerator::checkDirection") 
				<< "position on Lower scintillator " 
				<< positionLower.second;
                 
                 return true;
        }
        edm::LogVerbatim("CtfSpecialSeedGenerator::checkDirection") 
		<< "scintillator not crossed in bounds: position on Upper scintillator " 
		<< positionUpper.second << " position on Lower scintillator " << positionLower.second;
        return false;
}


