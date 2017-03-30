#include "RecoTracker/TkSeedGenerator/plugins/MultiHitGeneratorFromChi2.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "RecoPixelVertexing/PixelTriplets/interface/ThirdHitPredictionFromCircle.h"
#include "RecoPixelVertexing/PixelTriplets/plugins/ThirdHitRZPrediction.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/ESInputTag.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "RecoPixelVertexing/PixelTriplets/plugins/ThirdHitCorrection.h"
#include "RecoTracker/TkHitPairs/interface/RecHitsSortedInPhi.h"

#include "RecoPixelVertexing/PixelTriplets/plugins/KDTreeLinkerAlgo.h" 
#include "RecoPixelVertexing/PixelTriplets/plugins/KDTreeLinkerTools.h"

#include "RecoPixelVertexing/PixelTrackFitting/interface/RZLine.h"
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

#include "TrackingTools/Records/interface/TransientRecHitRecord.h"


#include "DataFormats/Math/interface/normalizedPhi.h"


#include "FWCore/Utilities/interface/isFinite.h"
#include "CommonTools/Utils/interface/DynArray.h"

#include <algorithm>
#include <iostream>
#include <vector>
#include <cmath>
#include <map>
#include <limits>

using namespace std;

typedef PixelRecoRange<float> Range;

namespace {
  struct LayerRZPredictions {
    ThirdHitRZPrediction<SimpleLineRZ> line;
  };
}

MultiHitGeneratorFromChi2::MultiHitGeneratorFromChi2(const edm::ParameterSet& cfg)
  : MultiHitGeneratorFromPairAndLayers(cfg),
    useFixedPreFiltering(cfg.getParameter<bool>("useFixedPreFiltering")),
    extraHitRZtolerance(cfg.getParameter<double>("extraHitRZtolerance")),//extra window in ThirdHitRZPrediction range 
    extraHitRPhitolerance(cfg.getParameter<double>("extraHitRPhitolerance")),//extra window in ThirdHitPredictionFromCircle range (divide by R to get phi) 
    extraZKDBox(cfg.getParameter<double>("extraZKDBox")),//extra windown in Z when building the KDTree box (used in barrel)
    extraRKDBox(cfg.getParameter<double>("extraRKDBox")),//extra windown in R when building the KDTree box (used in endcap)
    extraPhiKDBox(cfg.getParameter<double>("extraPhiKDBox")),//extra windown in Phi when building the KDTree box
    fnSigmaRZ(cfg.getParameter<double>("fnSigmaRZ")),//this multiplies the max hit error on the layer when building the KDTree box
    chi2VsPtCut(cfg.getParameter<bool>("chi2VsPtCut")),
    maxChi2(cfg.getParameter<double>("maxChi2")),
    refitHits(cfg.getParameter<bool>("refitHits")),
    filterName_(cfg.getParameter<std::string>("ClusterShapeHitFilterName")),
    builderName_(cfg.existsAs<std::string>("TTRHBuilder") ? cfg.getParameter<std::string>("TTRHBuilder") : std::string("WithTrackAngle")),
    useSimpleMF_(false),
    mfName_("")
{    
  if (useFixedPreFiltering)
    dphi = cfg.getParameter<double>("phiPreFiltering");
  if (chi2VsPtCut) {
    pt_interv = cfg.getParameter<std::vector<double> >("pt_interv");
    chi2_cuts = cfg.getParameter<std::vector<double> >("chi2_cuts");    
  }  
#ifdef EDM_ML_DEBUG
  detIdsToDebug = cfg.getParameter<std::vector<int> >("detIdsToDebug");
  //if (detIdsToDebug.size()<3) //fixme
#else
  detIdsToDebug.push_back(0);
  detIdsToDebug.push_back(0);
  detIdsToDebug.push_back(0);
#endif
  // 2014/02/11 mia:
  // we should get rid of the boolean parameter useSimpleMF,
  // and use only a string magneticField [instead of SimpleMagneticField]
  // or better an edm::ESInputTag (at the moment HLT does not handle ESInputTag)
  if (cfg.exists("SimpleMagneticField")) {
    useSimpleMF_ = true;
    mfName_ = cfg.getParameter<std::string>("SimpleMagneticField");
  }
  filter = 0;
  bfield = 0;
  nomField = -1.;
}

MultiHitGeneratorFromChi2::~MultiHitGeneratorFromChi2() {}


void MultiHitGeneratorFromChi2::fillDescriptions(edm::ParameterSetDescription& desc) {
  MultiHitGeneratorFromPairAndLayers::fillDescriptions(desc);

  // fixed phi filtering
  desc.add<bool>("useFixedPreFiltering", false);
  desc.add<double>("phiPreFiltering", 0.3);

  // box properties
  desc.add<double>("extraHitRPhitolerance", 0);
  desc.add<double>("extraHitRZtolerance", 0);
  desc.add<double>("extraZKDBox", 0.2);
  desc.add<double>("extraRKDBox", 0.2);
  desc.add<double>("extraPhiKDBox", 0.005);
  desc.add<double>("fnSigmaRZ", 2.0);

  // refit&filter hits
  desc.add<bool>("refitHits", true);
  desc.add<std::string>("ClusterShapeHitFilterName", "ClusterShapeHitFilter");
  desc.add<std::string>("TTRHBuilder", "WithTrackAngle");

  // chi2 cuts
  desc.add<double>("maxChi2", 5.0);
  desc.add<bool>("chi2VsPtCut", true);
  desc.add<std::vector<double> >("pt_interv", std::vector<double>{{0.4,0.7,1.0,2.0}});
  desc.add<std::vector<double> >("chi2_cuts", std::vector<double>{{3.0,4.0,5.0,5.0}});

  // debugging
  desc.add<std::vector<int> >("detIdsToDebug", std::vector<int>{{0,0,0}});
}


void MultiHitGeneratorFromChi2::initES(const edm::EventSetup& es) 
{

  edm::ESHandle<MagneticField> bfield_h;
  if (useSimpleMF_) es.get<IdealMagneticFieldRecord>().get(mfName_, bfield_h);  
  else es.get<IdealMagneticFieldRecord>().get(bfield_h);
  bfield = bfield_h.product();
  nomField = bfield->nominalValue();

  if(refitHits)
  {
      edm::ESHandle<ClusterShapeHitFilter> filterHandle_;
      es.get<CkfComponentsRecord>().get(filterName_, filterHandle_);
      filter = filterHandle_.product();
      
      edm::ESHandle<TransientTrackingRecHitBuilder> builderH;
      es.get<TransientRecHitRecord>().get(builderName_, builderH);
      builder = (TkTransientTrackingRecHitBuilder const *)(builderH.product());
      cloner = (*builder).cloner();
  }
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
					const edm::EventSetup& es,
					SeedingLayerSetsHits::SeedingLayerSet pairLayers,
					std::vector<SeedingLayerSetsHits::SeedingLayer> thirdLayers)
{
  LogDebug("MultiHitGeneratorFromChi2") << "pair: " << thePairGenerator->innerLayer(pairLayers).name() << "+" <<  thePairGenerator->outerLayer(pairLayers).name() << " 3rd lay size: " << thirdLayers.size();

  auto const & doublets = thePairGenerator->doublets(region,ev,es, pairLayers);
  LogTrace("MultiHitGeneratorFromChi2") << "";
  if (doublets.empty()) {
    //  LogDebug("MultiHitGeneratorFromChi2") << "empy pairs";                                                      
    return;
  }

  assert(theLayerCache);
  hitSets(region, result, ev, es, doublets, thirdLayers, *theLayerCache, cache);
}

void MultiHitGeneratorFromChi2::hitSets(const TrackingRegion& region, OrderedMultiHits& result,
                                        const edm::Event& ev, const edm::EventSetup& es,
                                        const HitDoublets& doublets,
                                        const std::vector<SeedingLayerSetsHits::SeedingLayer>& thirdLayers,
                                        LayerCacheType& layerCache,
                                        cacheHits& refittedHitStorage) {
  int size = thirdLayers.size();
  const RecHitsSortedInPhi * thirdHitMap[size];
  vector<const DetLayer *> thirdLayerDetLayer(size,0);
  for (int il=0; il<size; ++il) 
    {
      thirdHitMap[il] = &layerCache(thirdLayers[il], region, es);

      thirdLayerDetLayer[il] = thirdLayers[il].detLayer();
    }
  hitSets(region, result, es, doublets, thirdHitMap, thirdLayerDetLayer, size, refittedHitStorage);
}

void MultiHitGeneratorFromChi2::hitTriplets(
					   const TrackingRegion& region, 
					   OrderedMultiHits & result,
					   const edm::EventSetup & es,
					   const HitDoublets & doublets,
					   const RecHitsSortedInPhi ** thirdHitMap,
					   const std::vector<const DetLayer *> & thirdLayerDetLayer,
					   const int nThirdLayers)
{
  hitSets(region, result, es, doublets, thirdHitMap, thirdLayerDetLayer, nThirdLayers, cache);
}

void MultiHitGeneratorFromChi2::hitSets(const TrackingRegion& region, OrderedMultiHits& result,
                                        const edm::EventSetup& es,
                                        const HitDoublets& doublets,
                                        const RecHitsSortedInPhi **thirdHitMap,
                                        const std::vector<const DetLayer *>& thirdLayerDetLayer,
                                        const int nThirdLayers,
                                        cacheHits& refittedHitStorage) {
  unsigned int debug_Id0 = detIdsToDebug[0];
  unsigned int debug_Id1 = detIdsToDebug[1];
  unsigned int debug_Id2 = detIdsToDebug[2];
  
  //gc: these are all the layers compatible with the layer pairs (as defined in the config file)


  //gc: initialize a KDTree per each 3rd layer
  std::vector<KDTreeNodeInfo<RecHitsSortedInPhi::HitIter> > layerTree; // re-used throughout
  std::vector<RecHitsSortedInPhi::HitIter> foundNodes; // re-used thoughout
  foundNodes.reserve(100);
  declareDynArray(KDTreeLinkerAlgo<RecHitsSortedInPhi::HitIter>,nThirdLayers, hitTree);
  declareDynArray(LayerRZPredictions, nThirdLayers, mapPred);
  float rzError[nThirdLayers]; //save maximum errors

  const float maxDelphi = region.ptMin() < 0.3f ? float(M_PI)/4.f : float(M_PI)/8.f; // FIXME move to config??
  const float maxphi = M_PI+maxDelphi, minphi = -maxphi; // increase to cater for any range
  const float safePhi = M_PI-maxDelphi; // sideband


  //gc: loop over each layer
  for(int il = 0; il < nThirdLayers; il++) {
    LogTrace("MultiHitGeneratorFromChi2") << "considering third layer: with hits: " << thirdHitMap[il]->all().second-thirdHitMap[il]->all().first;
    const DetLayer *layer = thirdLayerDetLayer[il];
    LayerRZPredictions &predRZ = mapPred[il];
    predRZ.line.initLayer(layer);
    predRZ.line.initTolerance(extraHitRZtolerance);

    //gc: now we take all hits in the layer and fill the KDTree    
    RecHitsSortedInPhi::Range hitRange = thirdHitMap[il]->all(); // Get iterators
    layerTree.clear();
    float minz=999999.0f, maxz= -minz; // Initialise to extreme values in case no hits
    float maxErr=0.0f;
    bool barrelLayer = (thirdLayerDetLayer[il]->location() == GeomDetEnumerators::barrel);
    if (hitRange.first != hitRange.second)
      { minz = barrelLayer? hitRange.first->hit()->globalPosition().z() : hitRange.first->hit()->globalPosition().perp();
	maxz = minz; //In case there's only one hit on the layer
	for (RecHitsSortedInPhi::HitIter hi=hitRange.first; hi != hitRange.second; ++hi)
	  { auto angle = hi->phi();
	    auto myz = barrelLayer? hi->hit()->globalPosition().z() : hi->hit()->globalPosition().perp();

            IfLogTrace(hi->hit()->rawId()==debug_Id2, "MultiHitGeneratorFromChi2") << "filling KDTree with hit in id=" << debug_Id2
                                                                                   << " with pos: " << hi->hit()->globalPosition()
                                                                                   << " phi=" << hi->hit()->globalPosition().phi()
                                                                                   << " z=" << hi->hit()->globalPosition().z()
                                                                                   << " r=" << hi->hit()->globalPosition().perp();
	    //use (phi,r) for endcaps rather than (phi,z)
	    if (myz < minz) { minz = myz;} else { if (myz > maxz) {maxz = myz;}}
	    auto myerr = barrelLayer? hi->hit()->errorGlobalZ(): hi->hit()->errorGlobalR();
	    if (myerr > maxErr) { maxErr = myerr;}
	    layerTree.push_back(KDTreeNodeInfo<RecHitsSortedInPhi::HitIter>(hi, angle, myz)); // save it
            // populate side-bands
            if (angle>safePhi) layerTree.push_back(KDTreeNodeInfo<RecHitsSortedInPhi::HitIter>(hi, angle-Geom::twoPi(), myz));
            else if (angle<-safePhi) layerTree.push_back(KDTreeNodeInfo<RecHitsSortedInPhi::HitIter>(hi, angle+Geom::twoPi(), myz));
	  }
      }
    KDTreeBox phiZ(minphi, maxphi, minz-0.01f, maxz+0.01f);  // declare our bounds
    //add fudge factors in case only one hit and also for floating-point inaccuracy
    hitTree[il].build(layerTree, phiZ); // make KDtree
    rzError[il] = maxErr; //save error
  }
  //gc: now we have initialized the KDTrees and we are out of the layer loop
  
  //gc: this sets the minPt of the triplet
  auto curv = PixelRecoUtilities::curvature(1. / region.ptMin(), es);

  LogTrace("MultiHitGeneratorFromChi2") << "doublet size=" << doublets.size() << std::endl;

  //gc: now we loop over all pairs
  for (std::size_t ip =0; ip!=doublets.size(); ip++) {

    int foundTripletsFromPair = 0;
    bool usePair = false;
    cacheHitPointer bestH2;
    float minChi2 = std::numeric_limits<float>::max();

    SeedingHitSet::ConstRecHitPointer oriHit0 =doublets.hit(ip,HitDoublets::inner);
    SeedingHitSet::ConstRecHitPointer oriHit1 =doublets.hit(ip,HitDoublets::outer);

    HitOwnPtr hit0(*oriHit0);
    HitOwnPtr hit1(*oriHit1);
    GlobalPoint gp0 = hit0->globalPosition();
    GlobalPoint gp1 = hit1->globalPosition();

#ifdef EDM_ML_DEBUG
    bool debugPair = oriHit0->rawId()==debug_Id0 && oriHit1->rawId()==debug_Id1;
#endif
    IfLogTrace(debugPair, "MultiHitGeneratorFromChi2") << endl << endl
                                                       << "found new pair with ids "<<debug_Id0<<" "<<debug_Id1<<" with pos: " << gp0 << " " << gp1;

    if (refitHits) {

      TrajectoryStateOnSurface tsos0, tsos1;
      assert(!hit0.isOwn()); assert(!hit1.isOwn());
#ifdef EDM_ML_DEBUG
      refit2Hits(hit0,hit1,tsos0,tsos1,region,nomField,debugPair);
#else
      refit2Hits(hit0,hit1,tsos0,tsos1,region,nomField,false);
#endif
      assert(hit0.isOwn()); assert(hit1.isOwn());

      //fixme add pixels
      bool passFilterHit0 = true;
      if (//hit0->geographicalId().subdetId() > 2
	  hit0->geographicalId().subdetId()==SiStripDetId::TIB 
	  || hit0->geographicalId().subdetId()==SiStripDetId::TID
	  //|| hit0->geographicalId().subdetId()==SiStripDetId::TOB
	  //|| hit0->geographicalId().subdetId()==SiStripDetId::TEC
	  ) {	
	const std::type_info &tid = typeid(*hit0->hit());
	if (tid == typeid(SiStripMatchedRecHit2D)) {
	  const SiStripMatchedRecHit2D* matchedHit = dynamic_cast<const SiStripMatchedRecHit2D *>(hit0->hit());
	  if (filter->isCompatible(DetId(matchedHit->monoId()), matchedHit->monoCluster(), tsos0.localMomentum())==0 ||
	      filter->isCompatible(DetId(matchedHit->stereoId()), matchedHit->stereoCluster(), tsos0.localMomentum())==0) passFilterHit0 = false;
	} else if (tid == typeid(SiStripRecHit2D)) {
	  const SiStripRecHit2D* recHit = dynamic_cast<const SiStripRecHit2D *>(hit0->hit());
	  if (filter->isCompatible(*recHit, tsos0.localMomentum())==0) passFilterHit0 = false;
	} else if (tid == typeid(ProjectedSiStripRecHit2D)) {
	  const ProjectedSiStripRecHit2D* precHit = dynamic_cast<const ProjectedSiStripRecHit2D *>(hit0->hit());
	  if (filter->isCompatible(precHit->originalHit(), tsos0.localMomentum())==0) passFilterHit0 = false;   //FIXME
	}
      }
      IfLogTrace(debugPair&&!passFilterHit0, "MultiHitGeneratorFromChi2") << "hit0 did not pass cluster shape filter";
      if (!passFilterHit0) continue;
      bool passFilterHit1 = true;
      if (//hit1->geographicalId().subdetId() > 2
	  hit1->geographicalId().subdetId()==SiStripDetId::TIB 
	  || hit1->geographicalId().subdetId()==SiStripDetId::TID
	  //|| hit1->geographicalId().subdetId()==SiStripDetId::TOB
	  //|| hit1->geographicalId().subdetId()==SiStripDetId::TEC
	  ) {	
	const std::type_info &tid = typeid(*hit1->hit());
	if (tid == typeid(SiStripMatchedRecHit2D)) {
	  const SiStripMatchedRecHit2D* matchedHit = dynamic_cast<const SiStripMatchedRecHit2D *>(hit1->hit());
	  if (filter->isCompatible(DetId(matchedHit->monoId()), matchedHit->monoCluster(), tsos1.localMomentum())==0 ||
	      filter->isCompatible(DetId(matchedHit->stereoId()), matchedHit->stereoCluster(), tsos1.localMomentum())==0) passFilterHit1 = false;
	} else if (tid == typeid(SiStripRecHit2D)) {
	  const SiStripRecHit2D* recHit = dynamic_cast<const SiStripRecHit2D *>(hit1->hit());
	  if (filter->isCompatible(*recHit, tsos1.localMomentum())==0) passFilterHit1 = false;
	} else if (tid == typeid(ProjectedSiStripRecHit2D)) {
	  const ProjectedSiStripRecHit2D* precHit = dynamic_cast<const ProjectedSiStripRecHit2D *>(hit1->hit());
	  if (filter->isCompatible(precHit->originalHit(), tsos1.localMomentum())==0) passFilterHit1 = false;  //FIXME
	}
      }
      IfLogTrace(debugPair&&!passFilterHit1, "MultiHitGeneratorFromChi2") << "hit1 did not pass cluster shape filter";
      if (!passFilterHit1) continue;

    } else {
      // not refit clone anyhow
      hit0.reset((BaseTrackerRecHit *)hit0->clone());
      hit1.reset((BaseTrackerRecHit *)hit1->clone());
    }

    //gc: create the RZ line for the pair
    SimpleLineRZ line(PixelRecoPointRZ(gp0.perp(),gp0.z()), PixelRecoPointRZ(gp1.perp(),gp1.z()));
    ThirdHitPredictionFromCircle predictionRPhi(gp0, gp1, extraHitRPhitolerance);

    auto toPos = std::signbit(gp1.z()-gp0.z());


    //gc: this is the curvature of the two hits assuming the region
    Range pairCurvature = predictionRPhi.curvature(region.originRBound());
    //gc: intersect not only returns a bool but may change pairCurvature to intersection with curv
    if (!intersect(pairCurvature, Range(-curv, curv))) {
      IfLogTrace(debugPair, "MultiHitGeneratorFromChi2") << "curvature cut: curv=" << curv 
                                                         << " gc=(" << pairCurvature.first << ", " << pairCurvature.second << ")";
      continue;
    }

    //gc: loop over all third layers compatible with the pair
    for(int il = 0; (il < nThirdLayers) & (!usePair); il++) {

      IfLogTrace(debugPair, "MultiHitGeneratorFromChi2") << "cosider layer:" << " for this pair. Location: " << thirdLayerDetLayer[il]->location();

      if (hitTree[il].empty()) {
	IfLogTrace(debugPair, "MultiHitGeneratorFromChi2") << "empty hitTree";
	continue; // Don't bother if no hits
      }

      cacheHitPointer bestL2;
      float chi2FromThisLayer = std::numeric_limits<float>::max();

      const DetLayer *layer = thirdLayerDetLayer[il];
      bool barrelLayer = layer->location() == GeomDetEnumerators::barrel;

      if ( (!barrelLayer) & (toPos != std::signbit(layer->position().z())) ) continue;


      LayerRZPredictions &predRZ = mapPred[il];
      predRZ.line.initPropagator(&line);
      
      //gc: this takes the z at R-thick/2 and R+thick/2 according to 
      //    the line from the two points and the adds the extra tolerance
      Range rzRange = predRZ.line();

      if (rzRange.first >= rzRange.second) {
	IfLogTrace(debugPair, "MultiHitGeneratorFromChi2") << "rzRange empty";
        continue;
      }
      //gc: check that rzRange is compatible with detector bounds
      //    note that intersect may change rzRange to intersection with bounds
      if (!intersect(rzRange, predRZ.line.detSize())) {// theDetSize = Range(-maxZ, maxZ); 
	IfLogTrace(debugPair, "MultiHitGeneratorFromChi2") << "rzRange and detector do not intersect";
	continue;
      }
      Range radius = barrelLayer ? predRZ.line.detRange() : rzRange;

      //gc: define the phi range of the hits
      Range phiRange;
      if (useFixedPreFiltering) { 
	//gc: in this case it takes as range the phi of the outer 
	//    hit +/- the phiPreFiltering value from cfg
        float phi0 = oriHit0->globalPosition().phi();
        phiRange = Range(phi0 - dphi, phi0 + dphi);
      } else {	
	//gc: predictionRPhi uses the cosine rule to find the phi of the 3rd point at radius, assuming the pairCurvature range [-c,+c]
	if (pairCurvature.first<0. && pairCurvature.second<0.) {
          radius.swap();
	} else if (pairCurvature.first>=0. && pairCurvature.second>=0.) {;}
        else {
	  radius.first=radius.second;
        }
        auto phi12 = predictionRPhi.phi(pairCurvature.first,radius.first);
	auto phi22 = predictionRPhi.phi(pairCurvature.second,radius.second);
        phi12 = normalizedPhi(phi12);
        phi22 = proxim(phi22,phi12);
	phiRange = Range(phi12,phi22); phiRange.sort();
      }
       
      float prmin=phiRange.min(), prmax=phiRange.max();
 
      if (prmax-prmin>maxDelphi) {
        auto prm = phiRange.mean();
        prmin = prm - 0.5f*maxDelphi;
        prmax = prm + 0.5f*maxDelphi;
      }


      //gc: this is the place where hits in the compatible region are put in the foundNodes
      using Hit=RecHitsSortedInPhi::Hit;
      foundNodes.clear(); // Now recover hits in bounding box...



      IfLogTrace(debugPair, "MultiHitGeneratorFromChi2") << "defining kd tree box";

      if (barrelLayer) {
	KDTreeBox phiZ(prmin-extraPhiKDBox, prmax+extraPhiKDBox,
		       rzRange.min()-fnSigmaRZ*rzError[il]-extraZKDBox,
		       rzRange.max()+fnSigmaRZ*rzError[il]+extraZKDBox);
	hitTree[il].search(phiZ, foundNodes);

	IfLogTrace(debugPair, "MultiHitGeneratorFromChi2") << "kd tree box bounds, phi: " << prmin-extraPhiKDBox <<","<< prmax+extraPhiKDBox
                                                           << " z: "<< rzRange.min()-fnSigmaRZ*rzError[il]-extraZKDBox <<","<<rzRange.max()+fnSigmaRZ*rzError[il]+extraZKDBox
                                                           << " rzRange: " << rzRange.min() <<","<<rzRange.max();

      } else {
	KDTreeBox phiR(prmin-extraPhiKDBox, prmax+extraPhiKDBox,
		       rzRange.min()-fnSigmaRZ*rzError[il]-extraRKDBox,
		       rzRange.max()+fnSigmaRZ*rzError[il]+extraRKDBox);
	hitTree[il].search(phiR, foundNodes);

	IfLogTrace(debugPair, "MultiHitGeneratorFromChi2") << "kd tree box bounds, phi: " << prmin-extraPhiKDBox <<","<< prmax+extraPhiKDBox
                                                           << " r: "<< rzRange.min()-fnSigmaRZ*rzError[il]-extraRKDBox <<","<<rzRange.max()+fnSigmaRZ*rzError[il]+extraRKDBox
                                                           << " rzRange: " << rzRange.min() <<","<<rzRange.max();
      }

      IfLogTrace(debugPair, "MultiHitGeneratorFromChi2") << "kd tree box size: " << foundNodes.size();
      

      //gc: now we loop over the hits in the box for this layer
      for (std::vector<RecHitsSortedInPhi::HitIter>::iterator ih = foundNodes.begin();
	   ih !=foundNodes.end() && !usePair; ++ih) {

	IfLogTrace(debugPair, "MultiHitGeneratorFromChi2") << endl << "triplet candidate";

	const RecHitsSortedInPhi::HitIter KDdata = *ih;

	SeedingHitSet::ConstRecHitPointer oriHit2 = KDdata->hit();
	cacheHitPointer hit2;

	if (refitHits) {//fixme

	  //fitting all 3 hits takes too much time... do it quickly only for 3rd hit
	  GlobalVector initMomentum(oriHit2->globalPosition() - gp1);
	  initMomentum *= (1./initMomentum.perp()); //set pT=1
	  GlobalTrajectoryParameters kine = GlobalTrajectoryParameters(oriHit2->globalPosition(), initMomentum, 1, &*bfield);
	  TrajectoryStateOnSurface state(kine,*oriHit2->surface());
	  hit2.reset((SeedingHitSet::RecHitPointer)(cloner(*oriHit2,state)));

	  //fixme add pixels
	  bool passFilterHit2 = true;
	  if (hit2->geographicalId().subdetId()==SiStripDetId::TIB 
	      || hit2->geographicalId().subdetId()==SiStripDetId::TID
	      // || hit2->geographicalId().subdetId()==SiStripDetId::TOB
	      // || hit2->geographicalId().subdetId()==SiStripDetId::TEC
	      ) {
	    const std::type_info &tid = typeid(*hit2->hit());
	    if (tid == typeid(SiStripMatchedRecHit2D)) {
	      const SiStripMatchedRecHit2D* matchedHit = dynamic_cast<const SiStripMatchedRecHit2D *>(hit2->hit());
	      if (filter->isCompatible(DetId(matchedHit->monoId()), matchedHit->monoCluster(), initMomentum)==0 ||
		  filter->isCompatible(DetId(matchedHit->stereoId()), matchedHit->stereoCluster(), initMomentum)==0) passFilterHit2 = false;
	    } else if (tid == typeid(SiStripRecHit2D)) {
	      const SiStripRecHit2D* recHit = dynamic_cast<const SiStripRecHit2D *>(hit2->hit());
	      if (filter->isCompatible(*recHit, initMomentum)==0) passFilterHit2 = false;
	    } else if (tid == typeid(ProjectedSiStripRecHit2D)) {
	      const ProjectedSiStripRecHit2D* precHit = dynamic_cast<const ProjectedSiStripRecHit2D *>(hit2->hit());
	      if (filter->isCompatible(precHit->originalHit(), initMomentum)==0) passFilterHit2 = false;
	    }
	  }
	  IfLogTrace(debugPair&&!passFilterHit2, "MultiHitGeneratorFromChi2") << "hit2 did not pass cluster shape filter";
	  if (!passFilterHit2) continue;
	  
	  // fitting all 3 hits takes too much time :-(
	  // TrajectoryStateOnSurface tsos0, tsos1, tsos2;
	  // refit3Hits(hit0,hit1,hit2,tsos0,tsos1,tsos2,nomField,debugPair);
	  // if (hit2->geographicalId().subdetId()==SiStripDetId::TIB 
	  //     // || hit2->geographicalId().subdetId()==SiStripDetId::TOB
	  //     // || hit2->geographicalId().subdetId()==SiStripDetId::TID
	  //     // || hit2->geographicalId().subdetId()==SiStripDetId::TEC
	  //     ) {
	  //   const std::type_info &tid = typeid(*hit2->hit());
	  //   if (tid == typeid(SiStripMatchedRecHit2D)) {
	  //     const SiStripMatchedRecHit2D* matchedHit = dynamic_cast<const SiStripMatchedRecHit2D *>(hit2->hit());
	  //     if (filter->isCompatible(DetId(matchedHit->monoId()), matchedHit->monoCluster(), tsos2.localMomentum())==0 ||
	  // 	      filter->isCompatible(DetId(matchedHit->stereoId()), matchedHit->stereoCluster(), tsos2.localMomentum())==0) continue;
	  //   } else if (tid == typeid(SiStripRecHit2D)) {
	  //     const SiStripRecHit2D* recHit = dynamic_cast<const SiStripRecHit2D *>(hit2->hit());
	  //     if (filter->isCompatible(*recHit, tsos2.localMomentum())==0) continue;
	  //   } else if (tid == typeid(ProjectedSiStripRecHit2D)) {
	  //     const ProjectedSiStripRecHit2D* precHit = dynamic_cast<const ProjectedSiStripRecHit2D *>(hit2->hit());
	  //     if (filter->isCompatible(precHit->originalHit(), tsos2.localMomentum())==0) continue;;
	  //   }
	  // }

	} else {
	  // not refit clone anyhow
	  hit2.reset((BaseTrackerRecHit*)oriHit2->clone());
	}

	//gc: add the chi2 cut
	std::array<GlobalPoint, 3> gp;
	std::array<GlobalError, 3> ge;
	std::array<bool, 3> bl;
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
	float chi2 = rzLine.chi2();

#ifdef EDM_ML_DEBUG
	bool debugTriplet = debugPair && hit2->rawId()==debug_Id2;
#endif
	IfLogTrace(debugTriplet, "MultiHitGeneratorFromChi2") << endl << "triplet candidate in debug id" << std::endl
                                                              << "hit in id="<<hit2->rawId()<<" (from KDTree) with pos: " << KDdata->hit()->globalPosition()
                                                              << " refitted: " << hit2->globalPosition() 
                                                              << " chi2: " << chi2;
	// should fix nan
	if ( (chi2 > maxChi2) | edm::isNotFinite(chi2) ) continue; 

	if (chi2VsPtCut) {

	  FastCircle theCircle(hit2->globalPosition(),hit1->globalPosition(),hit0->globalPosition());
	  float tesla0 = 0.1*nomField;
	  float rho = theCircle.rho();
	  float cm2GeV = 0.01 * 0.3*tesla0;
	  float pt = cm2GeV * rho;
	  IfLogTrace(debugTriplet, "MultiHitGeneratorFromChi2") << "triplet pT=" << pt;
	  if (pt<region.ptMin()) continue;
	  
	  if (chi2_cuts.size()==4) {
	    if ( ( pt>pt_interv[0] && pt<=pt_interv[1] && chi2 > chi2_cuts[0] ) ||
		 ( pt>pt_interv[1] && pt<=pt_interv[2] && chi2 > chi2_cuts[1] ) ||
		 ( pt>pt_interv[2] && pt<=pt_interv[3] && chi2 > chi2_cuts[2] ) ||
		 ( pt>pt_interv[3] && chi2 > chi2_cuts[3] ) ) continue;		
	  }
	  
	  // apparently this takes too much time...
	  // 	  if (chi2_cuts.size()>1) {	    
	  // 	    int ncuts = chi2_cuts.size();
	  // 	    if ( pt<=pt_interv[0] && chi2 > chi2_cuts[0] ) continue; 
	  // 	    bool pass = true;
	  // 	    for (int icut=1; icut<ncuts-1; icut++){
	  // 	      if ( pt>pt_interv[icut-1] && pt<=pt_interv[icut] && chi2 > chi2_cuts[icut] ) pass=false;
	  // 	    }
	  // 	    if (!pass) continue;
	  // 	    if ( pt>pt_interv[ncuts-2] && chi2 > chi2_cuts[ncuts-1] ) continue; 	    
	  // 	    if (hit0->rawId()==debug_Id0 && hit1->rawId()==debug_Id1 && hit2->rawId()==debug_Id2) {
	  // 	      LogTrace("MultiHitGeneratorFromChi2") << "triplet passed chi2 vs pt cut" << std::endl;
	  // 	    }
	  // 	  } 
	  
	}
	
	if (theMaxElement!=0 && result.size() >= theMaxElement) {
	  result.clear();
	  edm::LogError("TooManyTriplets")<<" number of triples exceed maximum. no triplets produced.";
	  return;
	}
	IfLogTrace(debugPair, "MultiHitGeneratorFromChi2") << "triplet made";
	//result.push_back(SeedingHitSet(hit0, hit1, hit2));
	/* no refit so keep only hit2
	assert(tripletFromThisLayer.empty());
	assert(hit0.isOwn()); assert(hit1.isOwn());assert(hit2.isOwn());
	tripletFromThisLayer.emplace_back(std::move(hit0));
	tripletFromThisLayer.emplace_back(std::move(hit1));
	tripletFromThisLayer.emplace_back(std::move(hit2));
	assert(hit0.isEmpty()); assert(hit1.isEmpty());assert(hit2.isEmpty());
	*/
	bestL2 = std::move(hit2);
	chi2FromThisLayer = chi2;
	foundTripletsFromPair++;
	if (foundTripletsFromPair>=2) {
	  usePair=true;
	  IfLogTrace(debugPair, "MultiHitGeneratorFromChi2") << "using pair";
	  break;
	}
      }//loop over hits in KDTree

      if (usePair) break;
      else {
	//if there is one triplet in more than one layer, try picking the one with best chi2
	if (chi2FromThisLayer<minChi2) {
	  bestH2 = std::move(bestL2);
	  minChi2 = chi2FromThisLayer;
	}
	/*
	else {
	  if (!bestH2 && foundTripletsFromPair>0)
	    LogTrace("MultiHitGeneratorFromChi2") << "what?? " <<  minChi2 << ' '  << chi2FromThisLayer;
	}
	*/
      }

    }//loop over layers

    if (foundTripletsFromPair==0) continue;

    //push back only (max) once per pair
    IfLogTrace(debugPair, "MultiHitGeneratorFromChi2") << "Done seed #" << result.size();
    if (usePair) result.push_back(SeedingHitSet(oriHit0, oriHit1)); 
    else { 
      assert(1==foundTripletsFromPair);
      assert(bestH2);
      result.emplace_back(&*hit0,&*hit1,&*bestH2); 
      assert(hit0.isOwn()); assert(hit1.isOwn());
      refittedHitStorage.emplace_back(const_cast<BaseTrackerRecHit*>(hit0.release()));
      refittedHitStorage.emplace_back(const_cast<BaseTrackerRecHit*>(hit1.release()));
      refittedHitStorage.emplace_back(std::move(bestH2));
      assert(hit0.empty()); assert(hit1.empty());assert(!bestH2);
    }
    // LogTrace("MultiHitGeneratorFromChi2") << (usePair ? "pair " : "triplet ") << minChi2 <<' ' << refittedHitStorage.size();


  }//loop over pairs
  LogTrace("MultiHitGeneratorFromChi2") << "triplet size=" << result.size();
}

void MultiHitGeneratorFromChi2::refit2Hits(HitOwnPtr & hit1,
					   HitOwnPtr & hit2,
					   TrajectoryStateOnSurface& state1,
					   TrajectoryStateOnSurface& state2,
					   const TrackingRegion& region, float nomField, bool isDebug) {

  //these need to be sorted in R
  GlobalPoint gp0 = region.origin();
  GlobalPoint gp1 = hit1->globalPosition();
  GlobalPoint gp2 = hit2->globalPosition();

  IfLogTrace(isDebug, "MultiHitGeneratorFromChi2") << "positions before refitting: " << hit1->globalPosition() << " " << hit2->globalPosition();

  FastCircle theCircle(gp2,gp1,gp0);
  GlobalPoint cc(theCircle.x0(),theCircle.y0(),0);
  float tesla0 = 0.1f*nomField;
  float rho = theCircle.rho();
  float cm2GeV = 0.01f * 0.3f*tesla0;
  float pt = cm2GeV * rho;

  GlobalVector vec20 = gp2-gp0;
  //if (isDebug) { cout << "vec20.eta=" << vec20.eta() << endl; }

  GlobalVector p0( gp0.y()-cc.y(), -gp0.x()+cc.x(), 0. );
  p0 = p0*pt/p0.perp();
  GlobalVector p1( gp1.y()-cc.y(), -gp1.x()+cc.x(), 0. );
  p1 = p1*pt/p1.perp();
  GlobalVector p2( gp2.y()-cc.y(), -gp2.x()+cc.x(), 0. );
  p2 = p2*pt/p2.perp();

  //check sign according to scalar product
  if ( (p0.x()*(gp1.x()-gp0.x())+p0.y()*(gp1.y()-gp0.y()) ) < 0 ) {
    p0*=-1.;
    p1*=-1.;
    p2*=-1.;
  }

  //now set z component
  auto zv = vec20.z()/vec20.perp();
  p0 = GlobalVector(p0.x(),p0.y(),p0.perp()*zv);
  p1 = GlobalVector(p1.x(),p1.y(),p1.perp()*zv);
  p2 = GlobalVector(p2.x(),p2.y(),p2.perp()*zv);

  //get charge from vectorial product
  TrackCharge q = 1;
  if ((gp1-cc).x()*p1.y() - (gp1-cc).y()*p1.x() > 0) q =-q;

  GlobalTrajectoryParameters kine1 = GlobalTrajectoryParameters(gp1, p1, q, &*bfield);
  state1 = TrajectoryStateOnSurface(kine1,*hit1->surface());
  hit1.reset((SeedingHitSet::RecHitPointer)(cloner(*hit1,state1)));

  GlobalTrajectoryParameters kine2 = GlobalTrajectoryParameters(gp2, p2, q, &*bfield);
  state2 = TrajectoryStateOnSurface(kine2,*hit2->surface());
  hit2.reset((SeedingHitSet::RecHitPointer)(cloner(*hit2,state2)));

  IfLogTrace(isDebug, "MultiHitGeneratorFromChi2") << "charge=" << q << std::endl
                                                   << "state1 pt=" << state1.globalMomentum().perp() << " eta=" << state1.globalMomentum().eta()  << " phi=" << state1.globalMomentum().phi() << std::endl
                                                   << "state2 pt=" << state2.globalMomentum().perp() << " eta=" << state2.globalMomentum().eta()  << " phi=" << state2.globalMomentum().phi() << std::endl
                                                   << "positions after refitting: " << hit1->globalPosition() << " " << hit2->globalPosition();
}

/*
void MultiHitGeneratorFromChi2::refit3Hits(HitOwnPtr & hit0,
					   HitOwnPtr & hit1,
					   HitOwnPtr & hit2,
					   TrajectoryStateOnSurface& state0,
					   TrajectoryStateOnSurface& state1,
					   TrajectoryStateOnSurface& state2,
					   float nomField, bool isDebug) {

  //these need to be sorted in R
  GlobalPoint gp0 = hit0->globalPosition();
  GlobalPoint gp1 = hit1->globalPosition();
  GlobalPoint gp2 = hit2->globalPosition();

  IfLogTrace(isDebug, "MultiHitGeneratorFromChi2") << "positions before refitting: " << hit0->globalPosition() << " " << hit1->globalPosition() << " " << hit2->globalPosition();
  }

  FastCircle theCircle(gp2,gp1,gp0);
  GlobalPoint cc(theCircle.x0(),theCircle.y0(),0);
  float tesla0 = 0.1*nomField;
  float rho = theCircle.rho();
  float cm2GeV = 0.01 * 0.3*tesla0;
  float pt = cm2GeV * rho;

  GlobalVector vec20 = gp2-gp0;
  //IfLogTrace(isDebug, "MultiHitGeneratorFromChi2") << "vec20.eta=" << vec20.eta();

  GlobalVector p0( gp0.y()-cc.y(), -gp0.x()+cc.x(), 0. );
  p0 = p0*pt/p0.perp();
  GlobalVector p1( gp1.y()-cc.y(), -gp1.x()+cc.x(), 0. );
  p1 = p1*pt/p1.perp();
  GlobalVector p2( gp2.y()-cc.y(), -gp2.x()+cc.x(), 0. );
  p2 = p2*pt/p2.perp();

  //check sign according to scalar product
  if ( (p0.x()*(gp1.x()-gp0.x())+p0.y()*(gp1.y()-gp0.y()) ) < 0 ) {
    p0*=-1.;
    p1*=-1.;
    p2*=-1.;
  }

  //now set z component
  p0 = GlobalVector(p0.x(),p0.y(),p0.perp()/tan(vec20.theta()));
  p1 = GlobalVector(p1.x(),p1.y(),p1.perp()/tan(vec20.theta()));
  p2 = GlobalVector(p2.x(),p2.y(),p2.perp()/tan(vec20.theta()));

  //get charge from vectorial product
  TrackCharge q = 1;
  if ((gp1-cc).x()*p1.y() - (gp1-cc).y()*p1.x() > 0) q =-q;

  GlobalTrajectoryParameters kine0 = GlobalTrajectoryParameters(gp0, p0, q, &*bfield);
  state0 = TrajectoryStateOnSurface(kine0,*hit0->surface());
  hit0 = hit0->clone(state0);

  GlobalTrajectoryParameters kine1 = GlobalTrajectoryParameters(gp1, p1, q, &*bfield);
  state1 = TrajectoryStateOnSurface(kine1,*hit1->surface());
  hit1 = hit1->clone(state1);

  GlobalTrajectoryParameters kine2 = GlobalTrajectoryParameters(gp2, p2, q, &*bfield);
  state2 = TrajectoryStateOnSurface(kine2,*hit2->surface());
  hit2 = hit2->clone(state2);

  IfLogTrace(isDebug, "MultiHitGeneratorFromChi2") << "charge=" << q << std::endl
                                                   << "state0 pt=" << state0.globalMomentum().perp() << " eta=" << state0.globalMomentum().eta()  << " phi=" << state0.globalMomentum().phi() << std::endl
                                                   << "state1 pt=" << state1.globalMomentum().perp() << " eta=" << state1.globalMomentum().eta()  << " phi=" << state1.globalMomentum().phi() << std::endl
                                                   << "state2 pt=" << state2.globalMomentum().perp() << " eta=" << state2.globalMomentum().eta()  << " phi=" << state2.globalMomentum().phi() << std::endl
                                                   << "positions after refitting: " << hit0->globalPosition() << " " << hit1->globalPosition() << " " << hit2->globalPosition();
}
*/

