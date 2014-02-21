#include "RecoTracker/TkSeedGenerator/plugins/MultiHitGeneratorFromChi2.h"

#include "RecoPixelVertexing/PixelTriplets/interface/ThirdHitPredictionFromCircle.h"
#include "RecoPixelVertexing/PixelTriplets/plugins/ThirdHitRZPrediction.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include <FWCore/Utilities/interface/ESInputTag.h>

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "RecoPixelVertexing/PixelTriplets/plugins/ThirdHitCorrection.h"
#include "RecoTracker/TkHitPairs/interface/RecHitsSortedInPhi.h"

#include "RecoPixelVertexing/PixelTriplets/plugins/KDTreeLinkerAlgo.h" 
#include "RecoPixelVertexing/PixelTriplets/plugins/KDTreeLinkerTools.h"

#include "RecoPixelVertexing/PixelTrackFitting/src/RZLine.h"
#include "RecoTracker/TkSeedGenerator/interface/FastHelix.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "RecoTracker/TkHitPairs/interface/HitPairGeneratorFromLayerPair.h"

#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/ProjectedSiStripRecHit2D.h"
#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"
#include "MagneticField/Engine/interface/MagneticField.h"

#include <algorithm>
#include <iostream>
#include <vector>
#include <cmath>
#include <map>
#include <limits>

using namespace std;

typedef PixelRecoRange<float> Range;

typedef ThirdHitPredictionFromCircle::HelixRZ HelixRZ;

namespace {
  struct LayerRZPredictions {
    ThirdHitRZPrediction<PixelRecoLineRZ> line;
    ThirdHitRZPrediction<HelixRZ> helix1, helix2;
  };
}

MultiHitGeneratorFromChi2::MultiHitGeneratorFromChi2(const edm::ParameterSet& cfg)
  : thePairGenerator(0)
  , theLayerCache(0)
  , useFixedPreFiltering (cfg.getParameter<bool>  ("useFixedPreFiltering")          )
  , extraHitRZtolerance  (cfg.getParameter<double>("extraHitRZtolerance")           )
  , extraHitRPhitolerance(cfg.getParameter<double>("extraHitRPhitolerance")         )
  , extraPhiKDBox        (cfg.getParameter<double>("extraPhiKDBox")                 )
  , fnSigmaRZ            (cfg.getParameter<double>("fnSigmaRZ")                     )
  , chi2VsPtCut          (cfg.getParameter<bool>  ("chi2VsPtCut")                   )
  , maxChi2              (cfg.getParameter<double>("maxChi2")                       )
  , refitHits            (cfg.getParameter<bool>  ("refitHits")                     )
  , debug                (cfg.getParameter<bool>  ("debug")                         )
  , filterName_          (cfg.getParameter<std::string>("ClusterShapeHitFilterName"))
  , useSimpleMF_         (false)
  , mfName_              ("")
{    
  theMaxElement=cfg.getParameter<unsigned int>("maxElement");
  if (useFixedPreFiltering)
    dphi = cfg.getParameter<double>("phiPreFiltering");
  if (chi2VsPtCut) {
    pt_interv = cfg.getParameter<std::vector<double> >("pt_interv");
    chi2_cuts = cfg.getParameter<std::vector<double> >("chi2_cuts");    
  }  
  if (debug) {
    detIdsToDebug = cfg.getParameter<std::vector<int> >("detIdsToDebug");
    //if (detIdsToDebug.size()<3) //fixme
  } else {
    detIdsToDebug.push_back(0);
    detIdsToDebug.push_back(0);
    detIdsToDebug.push_back(0);
  }
  // 2014/02/11 mia:
  // we should get rid of the boolean parameter useSimpleMF,
  // and use only a string magneticField [instead of SimpleMagneticField]
  // or better an edm::ESInputTag (at the moment HLT does not handle ESInputTag)
  if (cfg.exists("SimpleMagneticField")) {
    useSimpleMF_ = true;
    mfName_ = cfg.getParameter<std::string>("SimpleMagneticField");
  }
  bfield = 0;
  nomField = -1.;
}

void MultiHitGeneratorFromChi2::init(const HitPairGenerator & pairs,
					 LayerCacheType *layerCache)
{
  thePairGenerator = pairs.clone();
  theLayerCache = layerCache;
}

void MultiHitGeneratorFromChi2::setSeedingLayers(SeedingLayerSetsHits::SeedingLayerSet pairLayers,
                                                 std::vector<SeedingLayerSetsHits::SeedingLayer> thirdLayers) {
  thePairGenerator->setSeedingLayers(pairLayers);
  theLayers = thirdLayers;
}

namespace {
  inline
  bool intersect(Range &range, const Range &second)
  {
    if (range.first > second.max() || range.second < second.min())
      return false;
    if (range.first < second.min())
      range.first = second.min();
    if (range.second > second.max())
      range.second = second.max();
    return range.first < range.second;
  }
}

void MultiHitGeneratorFromChi2::hitSets(const TrackingRegion& region, 
					OrderedMultiHits & result,
					const edm::Event & ev,
					const edm::EventSetup& es)
{ 

  //gc: why is this here and not in some initialization???
  edm::ESHandle<TrackerGeometry> tracker;
  es.get<TrackerDigiGeometryRecord>().get(tracker);
  if (nomField<0 && bfield == 0) {
    edm::ESHandle<MagneticField> bfield_h;
    es.get<IdealMagneticFieldRecord>().get(mfName_, bfield_h);  
    //    edm::ESInputTag mfESInputTag(mfName_);
    //    es.get<IdealMagneticFieldRecord>().get(mfESInputTag, bfield_h);  
    bfield = bfield_h.product();
    nomField = bfield->nominalValue();
  }

  edm::ESHandle<ClusterShapeHitFilter> filterHandle_;
  es.get<CkfComponentsRecord>().get(filterName_, filterHandle_);
  filter = filterHandle_.product();

  //Retrieve tracker topology from geometry
  //edm::ESHandle<TrackerTopology> tTopoHand;
  //es.get<IdealGeometryRecord>().get(tTopoHand);
  //const TrackerTopology *tTopo=tTopoHand.product();

  if (debug) cout << "pair: " << ((HitPairGeneratorFromLayerPair*) thePairGenerator)->innerLayer().name() << "+" <<  ((HitPairGeneratorFromLayerPair*) thePairGenerator)->outerLayer().name() << " 3rd lay size: " << theLayers.size() << endl;

  //gc: first get the pairs
  OrderedHitPairs pairs;
  pairs.reserve(30000);
  thePairGenerator->hitPairs(region,pairs,ev,es);
  if (pairs.empty()) {
    //cout << "empy pairs" << endl;
    return;
  }
  
  //gc: these are all the layers compatible with the layer pairs (as defined in the config file)
  int size = theLayers.size();

  unsigned int debug_Id0 = detIdsToDebug[0];//402664068;
  unsigned int debug_Id1 = detIdsToDebug[1];//402666628;
  unsigned int debug_Id2 = detIdsToDebug[2];//402669320;//470049160;

  //gc: initialize a KDTree per each 3rd layer
  std::vector<KDTreeNodeInfo<RecHitsSortedInPhi::HitIter> > layerTree; // re-used throughout
  std::vector<RecHitsSortedInPhi::HitIter> foundNodes; // re-used thoughout
  foundNodes.reserve(100);
  KDTreeLinkerAlgo<RecHitsSortedInPhi::HitIter> hitTree[size];
  float rzError[size]; //save maximum errors
  double maxphi = Geom::twoPi(), minphi = -maxphi; //increase to cater for any range

  //map<const DetLayer*, LayerRZPredictions> mapPred;//gc
  map<std::string, LayerRZPredictions> mapPred;//need to use the name as map key since we may have more than one SeedingLayer per DetLayer (e.g. TEC and OTEC)
  const RecHitsSortedInPhi * thirdHitMap[size];//gc: this comes from theLayerCache

  //gc: loop over each layer
  for(int il = 0; il < size; il++) {
    thirdHitMap[il] = &(*theLayerCache)(theLayers[il], region, ev, es);
    if (debug) cout << "considering third layer: " << theLayers[il].name() << " with hits: " << thirdHitMap[il]->all().second-thirdHitMap[il]->all().first << endl;
    const DetLayer *layer = theLayers[il].detLayer();
    LayerRZPredictions &predRZ = mapPred[theLayers[il].name()];
    predRZ.line.initLayer(layer);
    predRZ.line.initTolerance(extraHitRZtolerance);

    //gc: now we take all hits in the layer and fill the KDTree    
    RecHitsSortedInPhi::Range hitRange = thirdHitMap[il]->all(); // Get iterators
    layerTree.clear();
    double minz=999999.0, maxz= -999999.0; // Initialise to extreme values in case no hits
    float maxErr=0.0f;
    bool barrelLayer = (theLayers[il].detLayer()->location() == GeomDetEnumerators::barrel);
    if (hitRange.first != hitRange.second)
      { minz = barrelLayer? hitRange.first->hit()->globalPosition().z() : hitRange.first->hit()->globalPosition().perp();
	maxz = minz; //In case there's only one hit on the layer
	for (RecHitsSortedInPhi::HitIter hi=hitRange.first; hi != hitRange.second; ++hi)
	  { double angle = hi->phi();
	    double myz = barrelLayer? hi->hit()->globalPosition().z() : hi->hit()->globalPosition().perp();

	    if (debug && hi->hit()->rawId()==debug_Id2) cout << "filling KDTree with hit in id=" << debug_Id2 
							     << " with pos: " << hi->hit()->globalPosition() 
							     << " phi=" << hi->hit()->globalPosition().phi() 
							     << " z=" << hi->hit()->globalPosition().z() 
							     << " r=" << hi->hit()->globalPosition().perp() 
							     << " trans: " << hi->hit()->transientHits()[0]->globalPosition() << " " 
							     << (hi->hit()->transientHits().size()>1 ? hi->hit()->transientHits()[1]->globalPosition() : GlobalPoint(0,0,0))
							     << endl;

	    //use (phi,r) for endcaps rather than (phi,z)
	    if (myz < minz) { minz = myz;} else { if (myz > maxz) {maxz = myz;}}
	    float myerr = barrelLayer? hi->hit()->errorGlobalZ(): hi->hit()->errorGlobalR();
	    if (myerr > maxErr) { maxErr = myerr;}
	    layerTree.push_back(KDTreeNodeInfo<RecHitsSortedInPhi::HitIter>(hi, angle, myz)); // save it
	    if (angle < 0)  // wrap all points in phi
	      { layerTree.push_back(KDTreeNodeInfo<RecHitsSortedInPhi::HitIter>(hi, angle+Geom::twoPi(), myz));}
	    else
	      { layerTree.push_back(KDTreeNodeInfo<RecHitsSortedInPhi::HitIter>(hi, angle-Geom::twoPi(), myz));}
	  }
      }
    KDTreeBox phiZ(minphi, maxphi, minz-0.01, maxz+0.01);  // declare our bounds
    //add fudge factors in case only one hit and also for floating-point inaccuracy
    hitTree[il].build(layerTree, phiZ); // make KDtree
    rzError[il] = maxErr; //save error
  }
  //gc: ok now we have initialized the KDTrees and we are out of the layer loop
  
  //gc: ok, this sets the minPt of the triplet
  double curv = PixelRecoUtilities::curvature(1. / region.ptMin(), es);

  if (debug) std::cout << "pair size=" << pairs.size() << std::endl;

  //gc: now we loop over all pairs
  for (OrderedHitPairs::const_iterator ip = pairs.begin(); ip != pairs.end(); ++ip) {

    int foundTripletsFromPair = 0;
    bool usePair = false;
    SeedingHitSet triplet;
    float minChi2 = std::numeric_limits<float>::max();

    TransientTrackingRecHit::ConstRecHitPointer hit0 = ip->inner();
    TransientTrackingRecHit::ConstRecHitPointer hit1 = ip->outer();

    GlobalPoint gp0 = hit0->globalPosition();//ip->inner()->globalPosition();
    GlobalPoint gp1 = hit1->globalPosition();//ip->outer()->globalPosition();

    if (refitHits) {//fixme
      GlobalVector pairMomentum(gp1 - gp0);
      pairMomentum *= (1./pairMomentum.perp()); //set pT=1
      GlobalTrajectoryParameters kinePair0 = GlobalTrajectoryParameters(hit0->globalPosition(), pairMomentum, 1, &*bfield);
      TrajectoryStateOnSurface statePair0(kinePair0,*hit0->surface());
      GlobalTrajectoryParameters kinePair1 = GlobalTrajectoryParameters(hit1->globalPosition(), pairMomentum, 1, &*bfield);
      TrajectoryStateOnSurface statePair1(kinePair1,*hit1->surface());
      hit0 = hit0->clone(statePair0);
      hit1 = hit1->clone(statePair1);

      if (/* hit0->geographicalId().subdetId() > 2 && (*/
	  hit0->geographicalId().subdetId()==SiStripDetId::TIB /*|| hit0->geographicalId().subdetId()==SiStripDetId::TOB)*/
	  ) {	
	const std::type_info &tid = typeid(*hit0->hit());
	if (tid == typeid(SiStripMatchedRecHit2D)) {
	  const SiStripMatchedRecHit2D* matchedHit = dynamic_cast<const SiStripMatchedRecHit2D *>(hit0->hit());
	  if (filter->isCompatible(DetId(matchedHit->monoId()), matchedHit->monoCluster(), pairMomentum)==0 ||
	      filter->isCompatible(DetId(matchedHit->stereoId()), matchedHit->stereoCluster(), pairMomentum)==0) continue;
	} else if (tid == typeid(SiStripRecHit2D)) {
	  const SiStripRecHit2D* recHit = dynamic_cast<const SiStripRecHit2D *>(hit0->hit());
	  if (filter->isCompatible(*recHit, pairMomentum)==0) continue;
	} else if (tid == typeid(ProjectedSiStripRecHit2D)) {
	  const ProjectedSiStripRecHit2D* precHit = dynamic_cast<const ProjectedSiStripRecHit2D *>(hit0->hit());
	  if (filter->isCompatible(precHit->originalHit(), pairMomentum)==0) continue;;
	}
      }

      if (/*hit1->geographicalId().subdetId() > 2 && (*/
	  hit1->geographicalId().subdetId()==SiStripDetId::TIB /*|| hit1->geographicalId().subdetId()==SiStripDetId::TOB)*/
	  ) {	
	const std::type_info &tid = typeid(*hit1->hit());
	if (tid == typeid(SiStripMatchedRecHit2D)) {
	  const SiStripMatchedRecHit2D* matchedHit = dynamic_cast<const SiStripMatchedRecHit2D *>(hit1->hit());
	  if (filter->isCompatible(DetId(matchedHit->monoId()), matchedHit->monoCluster(), pairMomentum)==0 ||
	      filter->isCompatible(DetId(matchedHit->stereoId()), matchedHit->stereoCluster(), pairMomentum)==0) continue;
	} else if (tid == typeid(SiStripRecHit2D)) {
	  const SiStripRecHit2D* recHit = dynamic_cast<const SiStripRecHit2D *>(hit1->hit());
	  if (filter->isCompatible(*recHit, pairMomentum)==0) continue;
	} else if (tid == typeid(ProjectedSiStripRecHit2D)) {
	  const ProjectedSiStripRecHit2D* precHit = dynamic_cast<const ProjectedSiStripRecHit2D *>(hit1->hit());
	  if (filter->isCompatible(precHit->originalHit(), pairMomentum)==0) continue;;
	}
      }


    }
    //const TransientTrackingRecHit::ConstRecHitPointer& hit0 = refitHits ? ip->inner()->clone(statePair0) : ip->inner();
    //const TransientTrackingRecHit::ConstRecHitPointer& hit1 = refitHits ? ip->outer()->clone(statePair1) : ip->outer();

    if (debug && ip->inner()->rawId()==debug_Id0 && ip->outer()->rawId()==debug_Id1) {
      cout << "found new pair with ids "<<debug_Id0<<" "<<debug_Id1<<" with pos: " << gp0 << " " << gp1 
    	   << " trans0: " << ip->inner()->transientHits()[0]->globalPosition() << " " << ip->inner()->transientHits()[1]->globalPosition()
    	   << " trans1: " << ip->outer()->transientHits()[0]->globalPosition() << " " << ip->outer()->transientHits()[1]->globalPosition()
    	   << endl;
    }

    //gc: create the RZ line for the pair
    PixelRecoLineRZ line(gp0, gp1);
    ThirdHitPredictionFromCircle predictionRPhi(gp0, gp1, extraHitRPhitolerance);

    //gc: this is the curvature of the two hits assuming the region
    Range generalCurvature = predictionRPhi.curvature(region.originRBound());
    if (!intersect(generalCurvature, Range(-curv, curv))) {
      if (debug && ip->inner()->rawId()==debug_Id0 && ip->outer()->rawId()==debug_Id1) std::cout << "curvature cut: curv=" << curv << " gc=(" << generalCurvature.first << ", " << generalCurvature.second << ")" << std::endl;
      continue;
    }

    //gc: loop over all third layers compatible with the pair
    for(int il = 0; il < size && !usePair; il++) {

      if (debug && ip->inner()->rawId()==debug_Id0 && ip->outer()->rawId()==debug_Id1) 
	cout << "cosider layer: " << theLayers[il].name() << " for this pair. Location: " << theLayers[il].detLayer()->location() << endl;

      if (hitTree[il].empty()) {
	if (debug && ip->inner()->rawId()==debug_Id0 && ip->outer()->rawId()==debug_Id1) {
	  cout << "empty hitTree" << endl;
	}
	continue; // Don't bother if no hits
      }

      SeedingHitSet tripletFromThisLayer;
      float chi2FromThisLayer = std::numeric_limits<float>::max();

      const DetLayer *layer = theLayers[il].detLayer();
      bool barrelLayer = layer->location() == GeomDetEnumerators::barrel;

      //gc: this is the curvature of the two hits assuming the region
      Range curvature = generalCurvature;
      
      //LayerRZPredictions &predRZ = mapPred.find(layer)->second;
      LayerRZPredictions &predRZ = mapPred.find(theLayers[il].name())->second;
      predRZ.line.initPropagator(&line);
      
      //gc: ok, this takes the z at R-thick/2 and R+thick/2 according to 
      //    the line from the two points and the adds the extra tolerance
      Range rzRange = predRZ.line();

      if (rzRange.first >= rzRange.second) {
	if (debug && ip->inner()->rawId()==debug_Id0 && ip->outer()->rawId()==debug_Id1) {
	  cout << "rzRange empty" << endl;
	}
        continue;
      }

      //gc: define the phi range of the hits
      Range phiRange;
      if (useFixedPreFiltering) { 
	//gc: in this case it takes as range the phi of the outer 
	//    hit +/- the phiPreFiltering value from cfg
        float phi0 = ip->outer()->globalPosition().phi();
        phiRange = Range(phi0 - dphi, phi0 + dphi);
      } else {
        Range radius;
	
        if (barrelLayer) {
	  //gc: this is R-thick/2 and R+thick/2
          radius = predRZ.line.detRange();
          if (!intersect(rzRange, predRZ.line.detSize())) {// theDetSize = Range(-maxZ, maxZ);
	    if (debug && ip->inner()->rawId()==debug_Id0 && ip->outer()->rawId()==debug_Id1) {
	      cout << "rzRange and detector do not intersect" << endl;
	    }
            continue;
	  }
        } else {
          radius = rzRange;
          if (!intersect(radius, predRZ.line.detSize())) {
	    if (debug && ip->inner()->rawId()==debug_Id0 && ip->outer()->rawId()==debug_Id1) {
	      cout << "rzRange and detector do not intersect" << endl;
	    }
            continue;
	  }
        }
	
	//gc: predictionRPhi uses the cosine rule to find the phi of the 3rd point at radius, assuming the curvature range [-c,+c]
	//not sure if it is really needed to do it both for radius.first and radius.second... maybe one can do it once and inflate a bit
        Range rPhi1 = predictionRPhi(curvature, radius.first);
        Range rPhi2 = predictionRPhi(curvature, radius.second);
        rPhi1.first  /= radius.first;
        rPhi1.second /= radius.first;
        rPhi2.first  /= radius.second;
        rPhi2.second /= radius.second;
        phiRange = mergePhiRanges(rPhi1, rPhi2);
	/*
	//test computing predictionRPhi only once
	float avgRad = (radius.first+radius.second)/2.;
	phiRange = predictionRPhi(curvature, avgRad);
	phiRange.first  = phiRange.first/avgRad  - 0.0;
	phiRange.second = phiRange.second/avgRad + 0.0;
	*/
      }
      
      //gc: this is the place where hits in the compatible region are put in the foundNodes
      typedef RecHitsSortedInPhi::Hit Hit;
      foundNodes.clear(); // Now recover hits in bounding box...
      float prmin=phiRange.min(), prmax=phiRange.max(); //get contiguous range
      if ((prmax-prmin) > Geom::twoPi())
	{ prmax=Geom::pi(); prmin = -Geom::pi();}
      else
	{ while (prmax>maxphi) { prmin -= Geom::twoPi(); prmax -= Geom::twoPi();}
	  while (prmin<minphi) { prmin += Geom::twoPi(); prmax += Geom::twoPi();}
	  // This needs range -twoPi to +twoPi to work
	}
      if (barrelLayer) {
	if (debug && hit0->rawId()==debug_Id0 && hit1->rawId()==debug_Id1) cout << "defining kd tree box" << endl;
	//gc: this is R-thick/2 and R+thick/2 (was already computed above!)
	Range detR = predRZ.line.detRange();
	//gc: it seems to me the same thing done here could be obtained from predRZ.line() which has already been computed
	Range regMin = predRZ.line(detR.min());
	Range regMax = predRZ.line(detR.max());
	if (regMax.min() < regMin.min()) { swap(regMax, regMin);}
	KDTreeBox phiZ(prmin-extraPhiKDBox, prmax+extraPhiKDBox,
		       regMin.min()-fnSigmaRZ*rzError[il],
		       regMax.max()+fnSigmaRZ*rzError[il]);

	if (debug && hit0->rawId()==debug_Id0 && hit1->rawId()==debug_Id1) cout << "kd tree box bounds, phi: " << prmin <<","<< prmax
										<< " z: "<< regMin.min()-fnSigmaRZ*rzError[il] <<","<<regMax.max()+fnSigmaRZ*rzError[il] 
										<< " detR: " << detR.min() <<","<<detR.max()
										<< " regMin: " << regMin.min() <<","<<regMin.max()
										<< " regMax: " << regMax.min() <<","<<regMax.max()
										<< endl;
	hitTree[il].search(phiZ, foundNodes);
      }
      else {
	KDTreeBox phiR(prmin-extraPhiKDBox, prmax+extraPhiKDBox,
		       rzRange.min()-fnSigmaRZ*rzError[il],
		       rzRange.max()+fnSigmaRZ*rzError[il]);
	hitTree[il].search(phiR, foundNodes);

	if (debug && hit0->rawId()==debug_Id0 && hit1->rawId()==debug_Id1) cout << "kd tree box bounds, phi: " << prmin <<","<< prmax
										<< " r: "<< rzRange.min()-fnSigmaRZ*rzError[il] <<","<<rzRange.max()+fnSigmaRZ*rzError[il] 
										<< " rzRange: " << rzRange.min() <<","<<rzRange.max()
										<< endl;

      }

      if (debug && hit0->rawId()==debug_Id0 && hit1->rawId()==debug_Id1) cout << "kd tree box size: " << foundNodes.size() << endl;
      

      //gc: now we loop over the hits in the box for this layer
      for (std::vector<RecHitsSortedInPhi::HitIter>::iterator ih = foundNodes.begin();
	   ih !=foundNodes.end() && !usePair; ++ih) {



	if (debug && hit0->rawId()==debug_Id0 && hit1->rawId()==debug_Id1) std::cout << "triplet candidate" << std::endl;

	const RecHitsSortedInPhi::HitIter KDdata = *ih;

	TransientTrackingRecHit::ConstRecHitPointer hit2 = KDdata->hit();
	if (refitHits) {//fixme
	  GlobalVector initMomentum(hit2->globalPosition() - gp1);
	  initMomentum *= (1./initMomentum.perp()); //set pT=1
	  if (/*hit2->geographicalId().subdetId() > 2 && (*/
	      hit2->geographicalId().subdetId()==SiStripDetId::TIB /*|| hit2->geographicalId().subdetId()==SiStripDetId::TOB)*/
	      ) {
	    const std::type_info &tid = typeid(*hit2->hit());
	    if (tid == typeid(SiStripMatchedRecHit2D)) {
	      const SiStripMatchedRecHit2D* matchedHit = dynamic_cast<const SiStripMatchedRecHit2D *>(hit2->hit());
	      if (filterHandle_->isCompatible(DetId(matchedHit->monoId()), matchedHit->monoCluster(), initMomentum)==0 ||
		      filterHandle_->isCompatible(DetId(matchedHit->stereoId()), matchedHit->stereoCluster(), initMomentum)==0) continue;
	    } else if (tid == typeid(SiStripRecHit2D)) {
	      const SiStripRecHit2D* recHit = dynamic_cast<const SiStripRecHit2D *>(hit2->hit());
	      if (filterHandle_->isCompatible(*recHit, initMomentum)==0) continue;
	    } else if (tid == typeid(ProjectedSiStripRecHit2D)) {
	      const ProjectedSiStripRecHit2D* precHit = dynamic_cast<const ProjectedSiStripRecHit2D *>(hit2->hit());
	      if (filterHandle_->isCompatible(precHit->originalHit(), initMomentum)==0) continue;;
	    }
	  }
	  GlobalTrajectoryParameters kine = GlobalTrajectoryParameters(hit2->globalPosition(), initMomentum, 1, &*bfield);
	  TrajectoryStateOnSurface state(kine,*hit2->surface());
	  hit2 = hit2->clone(state);
	}
	//const TransientTrackingRecHit::ConstRecHitPointer& hit2 = refitHits ? KDdata->hit()->clone(state) : KDdata->hit();

	//gc: try to add the chi2 cut
	vector<GlobalPoint> gp(3);
	vector<GlobalError> ge(3);
	vector<bool> bl(3);
	gp[0] = hit0->globalPosition();
	ge[0] = hit0->globalPositionError();
	int subid0 = hit0->geographicalId().subdetId();
	bl[0] = (subid0 == StripSubdetector::TIB || subid0 == StripSubdetector::TOB || subid0 == (int) PixelSubdetector::PixelBarrel);
	gp[1] = hit1->globalPosition();
	ge[1] = hit1->globalPositionError();
	int subid1 = hit1->geographicalId().subdetId();
	bl[1] = (subid1 == StripSubdetector::TIB || subid1 == StripSubdetector::TOB || subid1 == (int) PixelSubdetector::PixelBarrel);
	gp[2] = hit2->globalPosition();
	ge[2] = hit2->globalPositionError();
	int subid2 = hit2->geographicalId().subdetId();
	bl[2] = (subid2 == StripSubdetector::TIB || subid2 == StripSubdetector::TOB || subid2 == (int) PixelSubdetector::PixelBarrel);
	RZLine rzLine(gp,ge,bl);
	float  cottheta, intercept, covss, covii, covsi;
	rzLine.fit(cottheta, intercept, covss, covii, covsi);
	float chi2 = rzLine.chi2(cottheta, intercept);

	if (debug && hit0->rawId()==debug_Id0 && hit1->rawId()==debug_Id1) {
	  if (hit2->rawId()==debug_Id2) {
	    std::cout << endl << "triplet candidate" << std::endl;
	    cout << "hit in id="<<debug_Id2<<" (from KDTree) with pos: " << KDdata->hit()->globalPosition()
		 << " refitted: " << hit2->globalPosition() 
		 << " trans2: " << hit2->transientHits()[0]->globalPosition() << " " << (hit2->transientHits().size()>1 ? hit2->transientHits()[1]->globalPosition() : GlobalPoint(0,0,0))
		 << " chi2: " << chi2
		 << endl;
	    //cout << state << endl;
	  }
	  //gc: try to add the chi2 cut OLD version
	  vector<float> r(3),z(3),errR(3);
	  r[0] = ip->inner()->globalPosition().perp();
	  z[0] = ip->inner()->globalPosition().z();
	  errR[0] = sqrt(ip->inner()->globalPositionError().cxx()+ip->inner()->globalPositionError().cyy());
	  r[1] = ip->outer()->globalPosition().perp();
	  z[1] = ip->outer()->globalPosition().z();
	  errR[1] = sqrt(ip->outer()->globalPositionError().cxx()+ip->outer()->globalPositionError().cyy());
	  r[2] = KDdata->hit()->globalPosition().perp();
	  z[2] = KDdata->hit()->globalPosition().z();
	  errR[2] = sqrt(KDdata->hit()->globalPositionError().cxx()+KDdata->hit()->globalPositionError().cxx());
	  RZLine oldLine(z,r,errR);
	  float  cottheta_old, intercept_old, covss_old, covii_old, covsi_old;
	  oldLine.fit(cottheta_old, intercept_old, covss_old, covii_old, covsi_old);
	  float chi2_old = oldLine.chi2(cottheta_old, intercept_old);
	  if (debug && hit0->rawId()==debug_Id0 && hit1->rawId()==debug_Id1 && hit2->rawId()==debug_Id2) {	
	    cout << "triplet with ids: " << hit0->geographicalId().rawId() << " " << hit1->geographicalId().rawId() << " " << hit2->geographicalId().rawId()
		 << " hitpos: " << gp[0] << " " << gp[1] << " " << gp[2] 
		 << " eta,phi: " << gp[0].eta() << "," << gp[0].phi() << " chi2: " << chi2  << " chi2_old: " << chi2_old << endl 
		 << "trans0: " << hit0->transientHits()[0]->globalPosition() << " " << hit0->transientHits()[1]->globalPosition() << endl 
		 << "trans1: " << hit1->transientHits()[0]->globalPosition() << " " << hit1->transientHits()[1]->globalPosition() << endl 
		 << "trans2: " << hit2->transientHits()[0]->globalPosition() << " " << (hit2->transientHits().size()>1 ? hit2->transientHits()[1]->globalPosition() : GlobalPoint(0,0,0))
		 << endl;
	  }
	}

	if (chi2 > maxChi2) continue; 

	if (chi2VsPtCut) {

	  //FastHelix helix = FastHelix(hit2->globalPosition(),hit1->globalPosition(),hit0->globalPosition(),nomField,bfield);
	  //if (helix.isValid()==0) continue;//fixme: check cases where helix fails
	  //this is to compute the status at the 3rd point
	  //FastCircle theCircle = helix.circle();

	  //GlobalPoint maxRhit2(hit2->globalPosition().x()+(hit2->globalPosition().x()>0 ? sqrt(hit2->globalPositionError().cxx()) : sqrt(hit2->globalPositionError().cxx())*(-1.)),
	  //hit2->globalPosition().y()+(hit2->globalPosition().y()>0 ? sqrt(hit2->globalPositionError().cyy()) : sqrt(hit2->globalPositionError().cyy())*(-1.)),
	  //hit2->globalPosition().z());
	  FastCircle theCircle(hit2->globalPosition(),hit1->globalPosition(),hit0->globalPosition());
	  float tesla0 = 0.1*nomField;
	  float rho = theCircle.rho();
	  float cm2GeV = 0.01 * 0.3*tesla0;
	  float pt = cm2GeV * rho;
	  if (debug && hit0->rawId()==debug_Id0 && hit1->rawId()==debug_Id1 && hit2->rawId()==debug_Id2) {
	    std::cout << "triplet pT=" << pt << std::endl;
	  }
	  if (pt<region.ptMin()) continue;
	  
	  if (chi2_cuts.size()>1) {	    
	    int ncuts = chi2_cuts.size();
	    if ( pt<=pt_interv[0] && chi2 > chi2_cuts[0] ) continue; 
	    bool pass = true;
	    for (int icut=1; icut<ncuts-1; icut++){
	      if ( pt>pt_interv[icut-1] && pt<=pt_interv[icut] && chi2 > chi2_cuts[icut] ) pass=false;
	    }
	    if (!pass) continue;
	    if ( pt>pt_interv[ncuts-2] && chi2 > chi2_cuts[ncuts-1] ) continue; 
	    
	    if (debug && hit0->rawId()==debug_Id0 && hit1->rawId()==debug_Id1 && hit2->rawId()==debug_Id2) {
	      std::cout << "triplet passed chi2 vs pt cut" << std::endl;
	    }
	  } 
	  
	}

	if (theMaxElement!=0 && result.size() >= theMaxElement) {
	  result.clear();
	  edm::LogError("TooManyTriplets")<<" number of triples exceed maximum. no triplets produced.";
	  return;
	}
	if (debug && hit0->rawId()==debug_Id0 && hit1->rawId()==debug_Id1 && hit2->rawId()==debug_Id2) std::cout << "triplet made" << std::endl;
	//result.push_back(SeedingHitSet(hit0, hit1, hit2)); 
	tripletFromThisLayer = SeedingHitSet(hit0, hit1, hit2);
	chi2FromThisLayer = chi2;
	foundTripletsFromPair++;
	if (foundTripletsFromPair>=2) {
	  usePair=true;
	  break;
	}
      }//loop over hits in KDTree

      if (usePair) break;
      else {
	//if there is one triplet in more than one layer, try picking the one with best chi2
	if (chi2FromThisLayer<minChi2) {
	  triplet = tripletFromThisLayer;
	  minChi2 = chi2FromThisLayer;
	}
      }

    }//loop over layers

    if (foundTripletsFromPair==0) continue;

    //push back only (max) once per pair
    if (usePair) result.push_back(SeedingHitSet(ip->inner(), ip->outer())); 
    else result.push_back(triplet); 

  }//loop over pairs
  if (debug) {
    std::cout << "triplet size=" << result.size() << std::endl;
  }
}

bool MultiHitGeneratorFromChi2::checkPhiInRange(float phi, float phi1, float phi2) const
{ while (phi > phi2) phi -= 2. * M_PI;
  while (phi < phi1) phi += 2. * M_PI;
  return phi <= phi2;
}  

std::pair<float, float>
MultiHitGeneratorFromChi2::mergePhiRanges(const std::pair<float, float> &r1,
					      const std::pair<float, float> &r2) const
{ float r2Min = r2.first;
  float r2Max = r2.second;
  while (r1.first - r2Min > +M_PI) r2Min += 2. * M_PI, r2Max += 2. * M_PI;
  while (r1.first - r2Min < -M_PI) r2Min -= 2. * M_PI, r2Max -= 2. * M_PI;
  //std::cout << "mergePhiRanges " << fabs(r1.first-r2Min) << " " <<  fabs(r1.second-r2Max) << endl;
  return std::make_pair(min(r1.first, r2Min), max(r1.second, r2Max));
}
