//
// Package:         RecoTracker/TkSeedGenerator
// Class:           GlobalPixelLessSeedGenerator
// 
// From CosmicSeedGenerator+SimpleCosmicBONSeeder, with changes by Giovanni
// to seed Cosmics with B != 0

#include "RecoTracker/SpecialSeedGenerators/interface/SimpleCosmicBONSeeder.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "FWCore/Utilities/interface/isFinite.h"
typedef TransientTrackingRecHit::ConstRecHitPointer SeedingHit;

#include <numeric>

using namespace std;
SimpleCosmicBONSeeder::SimpleCosmicBONSeeder(edm::ParameterSet const& conf) : 
  conf_(conf),
  theLsb(conf.getParameter<edm::ParameterSet>("TripletsPSet")),
  writeTriplets_(conf.getParameter<bool>("writeTriplets")),
  seedOnMiddle_(conf.existsAs<bool>("seedOnMiddle") ? conf.getParameter<bool>("seedOnMiddle") : false),
  rescaleError_(conf.existsAs<double>("rescaleError") ? conf.getParameter<double>("rescaleError") : 1.0),
  tripletsVerbosity_(conf.getParameter<edm::ParameterSet>("TripletsPSet").getUntrackedParameter<uint32_t>("debugLevel",0)),
  seedVerbosity_(conf.getUntrackedParameter<uint32_t>("seedDebugLevel",0)),
  helixVerbosity_(conf.getUntrackedParameter<uint32_t>("helixDebugLevel",0)),
  check_(conf.getParameter<edm::ParameterSet>("ClusterCheckPSet")),
  maxTriplets_(conf.getParameter<int32_t>("maxTriplets")),
  maxSeeds_(conf.getParameter<int32_t>("maxSeeds"))
{
  edm::ParameterSet regionConf = conf_.getParameter<edm::ParameterSet>("RegionPSet");
  float ptmin        = regionConf.getParameter<double>("ptMin");
  float originradius = regionConf.getParameter<double>("originRadius");
  float halflength   = regionConf.getParameter<double>("originHalfLength");
  float originz      = regionConf.getParameter<double>("originZPosition");
  region_ = GlobalTrackingRegion(ptmin, originradius, halflength, originz);
  pMin_   = regionConf.getParameter<double>("pMin");

  builderName = conf_.getParameter<std::string>("TTRHBuilder");   

  //***top-bottom
  positiveYOnly=conf_.getParameter<bool>("PositiveYOnly");
  negativeYOnly=conf_.getParameter<bool>("NegativeYOnly");
  //***

  produces<TrajectorySeedCollection>();
  if (writeTriplets_) produces<edm::OwnVector<TrackingRecHit> >("cosmicTriplets");

  layerTripletNames_ = conf_.getParameter<edm::ParameterSet>("TripletsPSet").getParameter<std::vector<std::string> >("layerList");

  if (conf.existsAs<edm::ParameterSet>("ClusterChargeCheck")) {
      edm::ParameterSet cccc = conf.getParameter<edm::ParameterSet>("ClusterChargeCheck");
      checkCharge_          = cccc.getParameter<bool>("checkCharge");
      matchedRecHitUsesAnd_ = cccc.getParameter<bool>("matchedRecHitsUseAnd");
      chargeThresholds_.resize(7,0);
      edm::ParameterSet ccct = cccc.getParameter<edm::ParameterSet>("Thresholds");
      chargeThresholds_[StripSubdetector::TIB] = ccct.getParameter<int32_t>("TIB");
      chargeThresholds_[StripSubdetector::TID] = ccct.getParameter<int32_t>("TID");
      chargeThresholds_[StripSubdetector::TOB] = ccct.getParameter<int32_t>("TOB");
      chargeThresholds_[StripSubdetector::TEC] = ccct.getParameter<int32_t>("TEC");
  } else {
      checkCharge_ = false;
  }
  if (conf.existsAs<edm::ParameterSet>("HitsPerModuleCheck")) {
      edm::ParameterSet hpmcc = conf.getParameter<edm::ParameterSet>("HitsPerModuleCheck");
      checkMaxHitsPerModule_  = hpmcc.getParameter<bool>("checkHitsPerModule");
      maxHitsPerModule_.resize(7,std::numeric_limits<int32_t>::max());
      edm::ParameterSet hpmct = hpmcc.getParameter<edm::ParameterSet>("Thresholds");
      maxHitsPerModule_[StripSubdetector::TIB] = hpmct.getParameter<int32_t>("TIB");
      maxHitsPerModule_[StripSubdetector::TID] = hpmct.getParameter<int32_t>("TID");
      maxHitsPerModule_[StripSubdetector::TOB] = hpmct.getParameter<int32_t>("TOB");
      maxHitsPerModule_[StripSubdetector::TEC] = hpmct.getParameter<int32_t>("TEC");
  } else {
      checkMaxHitsPerModule_ = false;
  }
  if (checkCharge_ || checkMaxHitsPerModule_) {
      goodHitsPerSeed_ = conf.getParameter<int32_t>("minimumGoodHitsInSeed"); 
  } else {
      goodHitsPerSeed_ = 0;
  }

}

// Functions that gets called by framework every event
void SimpleCosmicBONSeeder::produce(edm::Event& ev, const edm::EventSetup& es)
{
  std::auto_ptr<TrajectorySeedCollection> output(new TrajectorySeedCollection());
  std::auto_ptr<edm::OwnVector<TrackingRecHit> > outtriplets(new edm::OwnVector<TrackingRecHit>());

  es.get<IdealMagneticFieldRecord>().get(magfield);
  if (magfield->inTesla(GlobalPoint(0,0,0)).mag() > 0.01) {
     size_t clustsOrZero = check_.tooManyClusters(ev);
     if (clustsOrZero) {
         edm::LogError("TooManyClusters") << "Found too many clusters (" << clustsOrZero << "), bailing out.\n";
     } else {
         init(es);
         bool tripletsOk = triplets(ev,es);
         if (tripletsOk) {

             bool seedsOk    = seeds(*output,es);
             if (!seedsOk) { }

             if (writeTriplets_) {
                 for (OrderedHitTriplets::const_iterator it = hitTriplets.begin(); it != hitTriplets.end(); ++it) {
                     const TrackingRecHit * hit1 = it->inner()->hit();    
                     const TrackingRecHit * hit2 = it->middle()->hit();
                     const TrackingRecHit * hit3 = it->outer()->hit();
                     outtriplets->push_back(hit1->clone());
                     outtriplets->push_back(hit2->clone());
                     outtriplets->push_back(hit3->clone());
                 }
             }
         }
         done();
     }
  }

  if (writeTriplets_) {
      ev.put(outtriplets, "cosmicTriplets");
  }
  ev.put(output);
}

void 
SimpleCosmicBONSeeder::init(const edm::EventSetup& iSetup)
{
    iSetup.get<TrackerDigiGeometryRecord>().get(tracker);
    iSetup.get<TransientRecHitRecord>().get(builderName,TTTRHBuilder);

    // FIXME: these should come from ES too!!
    thePropagatorAl = new PropagatorWithMaterial(alongMomentum,0.1057,&(*magfield) );
    thePropagatorOp = new PropagatorWithMaterial(oppositeToMomentum,0.1057,&(*magfield) );
    theUpdator      = new KFUpdator();

}

struct HigherInnerHit {
    bool operator()(const OrderedHitTriplet &trip1, const OrderedHitTriplet &trip2) const {
        //FIXME: inner gives a SEGV
#if 0
        //const TransientTrackingRecHit::ConstRecHitPointer &ihit1 = trip1.inner();
        //const TransientTrackingRecHit::ConstRecHitPointer &ihit2 = trip2.inner();
        const TransientTrackingRecHit::ConstRecHitPointer &ihit1 = trip1.middle();
        const TransientTrackingRecHit::ConstRecHitPointer &ihit2 = trip2.middle();
        const TransientTrackingRecHit::ConstRecHitPointer &ohit1 = trip1.outer();
        const TransientTrackingRecHit::ConstRecHitPointer &ohit2 = trip2.outer();
#endif
        TransientTrackingRecHit::ConstRecHitPointer ihit1 = trip1.inner();
        TransientTrackingRecHit::ConstRecHitPointer ihit2 = trip2.inner();
        TransientTrackingRecHit::ConstRecHitPointer ohit1 = trip1.outer();
        TransientTrackingRecHit::ConstRecHitPointer ohit2 = trip2.outer();
        float iy1 = ihit1->globalPosition().y();
        float oy1 = ohit1->globalPosition().y();
        float iy2 = ihit2->globalPosition().y();
        float oy2 = ohit2->globalPosition().y();
        if (oy1 - iy1 > 0) { // 1 Downgoing
            if (oy2 - iy2 > 0) { // 2 Downgoing
                // sort by inner, or by outer if inners are the same
                return (iy1 != iy2 ? (iy1 > iy2) : (oy1 > oy2));
            } else return true; // else prefer downgoing
        } else if (oy2 - iy2 > 0) {
            return false; // prefer downgoing
        } else { // both upgoing
            // sort by inner, or by outer
            return (iy1 != iy2 ? (iy1 < iy2) : (oy1 < oy2));
        }
    }
};

bool SimpleCosmicBONSeeder::triplets(const edm::Event& e, const edm::EventSetup& es) {
    using namespace ctfseeding;

    hitTriplets.clear();
    hitTriplets.reserve(0);
    SeedingLayerSets lss = theLsb.layers(es);
    SeedingLayerSets::const_iterator iLss;

    double minRho = region_.ptMin() / ( 0.003 * magfield->inTesla(GlobalPoint(0,0,0)).z() );

    for (iLss = lss.begin(); iLss != lss.end(); iLss++){
        SeedingLayers ls = *iLss;
        if (ls.size() != 3){
            throw cms::Exception("CtfSpecialSeedGenerator") << "You are using " << ls.size() <<" layers in set instead of 3 ";
        }

        /// ctfseeding SeedinHits and their iterators
        std::vector<SeedingHit> innerHits  = region_.hits(e, es, &ls[0]);
        std::vector<SeedingHit> middleHits = region_.hits(e, es, &ls[1]);
        std::vector<SeedingHit> outerHits  = region_.hits(e, es, &ls[2]);
        std::vector<SeedingHit>::const_iterator iOuterHit,iMiddleHit,iInnerHit;

        if (tripletsVerbosity_ > 0) {
            std::cout << "GenericTripletGenerator iLss = " << layerTripletNames_[iLss - lss.begin()]
                    << " (" << (iLss - lss.begin()) << "): # = " 
                    << innerHits.size() << "/" << middleHits.size() << "/" << outerHits.size() << std::endl;
        }

        /// Transient Tracking RecHits
        typedef TransientTrackingRecHit::RecHitPointer TTRH;
        std::vector<TTRH> innerTTRHs, middleTTRHs, outerTTRHs;

        /// Checks on the cluster charge and on noisy modules
        std::vector<bool> innerOk( innerHits.size(),  true);
        std::vector<bool> middleOk(middleHits.size(), true);
        std::vector<bool> outerOk( outerHits.size(),  true);

        size_t sizBefore = hitTriplets.size();
        /// Now actually filling in the charges for all the clusters
        int idx = 0;
        for (iOuterHit = outerHits.begin(), idx = 0; iOuterHit != outerHits.end(); ++idx, ++iOuterHit){
            outerTTRHs.push_back(ls[2].hitBuilder()->build((**iOuterHit).hit()));
            if (checkCharge_ && !checkCharge(outerTTRHs.back()->hit())) outerOk[idx] = false;
        }
        for (iMiddleHit = middleHits.begin(), idx = 0; iMiddleHit != middleHits.end(); ++idx, ++iMiddleHit){
            middleTTRHs.push_back(ls[1].hitBuilder()->build((**iMiddleHit).hit()));
            if (checkCharge_ && !checkCharge(middleTTRHs.back()->hit())) middleOk[idx] = false;
        }
        for (iInnerHit = innerHits.begin(), idx = 0; iInnerHit != innerHits.end(); ++idx, ++iInnerHit){
            innerTTRHs.push_back(ls[0].hitBuilder()->build((**iInnerHit).hit()));
            if (checkCharge_ && !checkCharge(innerTTRHs.back()->hit())) innerOk[idx] = false;
        }
        if (checkMaxHitsPerModule_) {
            checkNoisyModules(innerTTRHs,  innerOk);
            checkNoisyModules(middleTTRHs, middleOk);
            checkNoisyModules(outerTTRHs,  outerOk);
        }

        for (iOuterHit = outerHits.begin(); iOuterHit != outerHits.end(); iOuterHit++){
            idx = iOuterHit - outerHits.begin();
            TTRH &       outerTTRH = outerTTRHs[idx];
            GlobalPoint  outerpos  = outerTTRH->globalPosition(); // this caches by itself
            bool         outerok   = outerOk[idx];
            if (outerok < goodHitsPerSeed_ - 2) {
                if (tripletsVerbosity_ > 2) 
                    std::cout << "Skipping at first hit: " << (outerok) << " < " << (goodHitsPerSeed_ - 2) << std::endl;
                continue; 
            }

            for (iMiddleHit = middleHits.begin(); iMiddleHit != middleHits.end(); iMiddleHit++){
                idx = iMiddleHit - middleHits.begin();
                TTRH &       middleTTRH = middleTTRHs[idx];
                GlobalPoint  middlepos  = middleTTRH->globalPosition(); // this caches by itself
                bool         middleok   = middleOk[idx];
                if (outerok+middleok < goodHitsPerSeed_ - 1) {
                    if (tripletsVerbosity_ > 2) 
                        std::cout << "Skipping at second hit: " << (outerok+middleok) << " < " << (goodHitsPerSeed_ - 1) << std::endl;
                    continue; 
                }

                for (iInnerHit = innerHits.begin(); iInnerHit != innerHits.end(); iInnerHit++){
                    idx = iInnerHit - innerHits.begin();
                    TTRH &       innerTTRH = innerTTRHs[idx];
                    GlobalPoint  innerpos  = innerTTRH->globalPosition(); // this caches by itself
                    bool         innerok   = innerOk[idx];
                    if (outerok+middleok+innerok < goodHitsPerSeed_) {
                        if (tripletsVerbosity_ > 2) 
                            std::cout << "Skipping at third hit: " << (outerok+middleok+innerok) << " < " << (goodHitsPerSeed_) << std::endl;
                        continue;
                    } 

		    //***top-bottom
		    if (positiveYOnly && (innerpos.y()<0 || middlepos.y()<0 || outerpos.y()<0
					  || outerpos.y() < innerpos.y()
					  ) ) continue;
		    if (negativeYOnly && (innerpos.y()>0 || middlepos.y()>0 || outerpos.y()>0
					  || outerpos.y() > innerpos.y()
					  ) ) continue;
		    //***
		    
                    if (tripletsVerbosity_ > 2) std::cout << "Trying seed with: " << innerpos << " + " << middlepos << " + " << outerpos << std::endl;
                    if (goodTriplet(innerpos,middlepos,outerpos,minRho)) {
                        OrderedHitTriplet oht(*iInnerHit,*iMiddleHit,*iOuterHit);
                        hitTriplets.push_back(oht);
                        if ((maxTriplets_ > 0) && (hitTriplets.size() > size_t(maxTriplets_))) {
                            hitTriplets.clear();                      // clear
                            //OrderedHitTriplets().swap(hitTriplets); // really clear   
                            edm::LogError("TooManyTriplets") << "Found too many triplets, bailing out.\n";
                            return false;
                        }
                        if (tripletsVerbosity_ > 3) {
                            std::cout << " accepted seed #" << (hitTriplets.size()-1) << " w/: " 
                                << innerpos << " + " << middlepos << " + " << outerpos << std::endl;
                        }
                        if (tripletsVerbosity_ == 2) {
                                std::cout << " good seed #" << (hitTriplets.size()-1) << " w/: "     
                                    << innerpos << " + " << middlepos << " + " << outerpos << std::endl;
                        }
                        if (tripletsVerbosity_ > 3 && (helixVerbosity_ > 0)) { // debug the momentum here too
                            pqFromHelixFit(innerpos,middlepos,outerpos,es); 
                        }
                    }
                }
            }
        }
        if ((tripletsVerbosity_ > 0) && (hitTriplets.size() > sizBefore)) {
            std::cout << "                        iLss = " << layerTripletNames_[iLss - lss.begin()]
                << " (" << (iLss - lss.begin()) << "): # = " 
                << innerHits.size() << "/" << middleHits.size() << "/" << outerHits.size() 
                << ": Found " << (hitTriplets.size() - sizBefore) << " seeds [running total: " << hitTriplets.size() << "]"
                << std::endl ;
        }

    }
    std::sort(hitTriplets.begin(),hitTriplets.end(),HigherInnerHit());
    return true;
}
bool SimpleCosmicBONSeeder::checkCharge(const TrackingRecHit *hit) const {
    DetId detid(hit->geographicalId());
    if (detid.det() != DetId::Tracker) return false; // should not happen
    int subdet = detid.subdetId(); 
    if (subdet < 3) { // pixel
        return true;
    } else {
        if (typeid(*hit) == typeid(SiStripMatchedRecHit2D)) {
            const SiStripMatchedRecHit2D *mhit = static_cast<const SiStripMatchedRecHit2D *>(hit);
            if (matchedRecHitUsesAnd_) {
                return checkCharge(mhit->monoHit(), subdet) && checkCharge(mhit->stereoHit(), subdet);
            } else {
                return checkCharge(mhit->monoHit(), subdet) || checkCharge(mhit->stereoHit(), subdet);
            }
        } else if (typeid(*hit) == typeid(SiStripRecHit2D)) {
            return checkCharge(static_cast<const SiStripRecHit2D &>(*hit), subdet);
        } else {
            return true;
        }
    }
}

// to be fixed to use OmniCluster
bool SimpleCosmicBONSeeder::checkCharge(const SiStripRecHit2D &hit, int subdetid) const {
    const SiStripCluster *clust = (hit.cluster().isNonnull() ?  hit.cluster().get() : hit.cluster_regional().get());
    int charge = std::accumulate(clust->amplitudes().begin(), clust->amplitudes().end(), int(0));
    if (tripletsVerbosity_ > 1) {
        std::cerr << "Hit on " << subdetid << ", charge = " << charge << ", threshold = " << chargeThresholds_[subdetid] 
                  << ", detid = " <<  hit.geographicalId().rawId() << ", firstStrip = " << clust->firstStrip() << std::endl;
    } else if ((tripletsVerbosity_ == 1) && (charge < chargeThresholds_[subdetid])) {
        std::cerr << "Hit on " << subdetid << ", charge = " << charge << ", threshold = " << chargeThresholds_[subdetid] 
                  << ", detid = " <<  hit.geographicalId().rawId() << ", firstStrip = " << clust->firstStrip() << std::endl;
    }
    return charge > chargeThresholds_[subdetid];
}

void SimpleCosmicBONSeeder::checkNoisyModules(const std::vector<TransientTrackingRecHit::RecHitPointer> &hits, std::vector<bool> &oks) const {
    typedef TransientTrackingRecHit::RecHitPointer TTRH;
    std::vector<TTRH>::const_iterator it = hits.begin(),  start = it,   end = hits.end();
    std::vector<bool>::iterator       ok = oks.begin(), okStart = ok;
    while (start < end) {
        DetId lastid = (*start)->geographicalId();
        for (it = start + 1; (it < end) && ((*it)->geographicalId() == lastid); ++it) {
            ++ok;
        }
        if ( (it - start) > maxHitsPerModule_[lastid.subdetId()] ) { 
            if (tripletsVerbosity_ > 0) {
                std::cerr << "SimpleCosmicBONSeeder: Marking noisy module " << lastid.rawId() << ", it has " << (it-start) << " rechits"
                          << " (threshold is " << maxHitsPerModule_[lastid.subdetId()] << ")" << std::endl;
            }
            std::fill(okStart,ok,false);
        } else if (tripletsVerbosity_ > 0) {
            if ( (it - start) > std::min(4,maxHitsPerModule_[lastid.subdetId()]/4) ) {
                std::cerr << "SimpleCosmicBONSeeder: Not marking noisy module " << lastid.rawId() << ", it has " << (it-start) << " rechits"
                          << " (threshold is " << maxHitsPerModule_[lastid.subdetId()] << ")" << std::endl;
            }
        }
        start = it; okStart = ok;
    }
}

bool SimpleCosmicBONSeeder::goodTriplet(const GlobalPoint &inner, const GlobalPoint & middle, const GlobalPoint & outer, const double &minRho) const {
    float dyOM = outer.y() - middle.y(), dyIM = inner.y() - middle.y();
    if ((dyOM * dyIM > 0) && (fabs(dyOM)>10) && (fabs(dyIM)>10)) {
        if (tripletsVerbosity_ > 2) std::cout << "  fail for non coherent dy" << std::endl;
        return false;
    }
    float dzOM = outer.z() - middle.z(), dzIM = inner.z() - middle.z();
    if ((dzOM * dzIM > 0) && (fabs(dzOM)>50) && (fabs(dzIM)>50)) {
        if (tripletsVerbosity_ > 2) std::cout << "  fail for non coherent dz" << std::endl;
        return false;
    }
    if (minRho > 0) {
        FastCircle theCircle(inner,middle,outer);
        if (theCircle.rho() < minRho) {
            if (tripletsVerbosity_ > 2) std::cout << "  fail for pt cut" << std::endl;
            return false;
        }
    }
    return true;
}

std::pair<GlobalVector,int>
SimpleCosmicBONSeeder::pqFromHelixFit(const GlobalPoint &inner, const GlobalPoint & middle, const GlobalPoint & outer, 
                                                         const edm::EventSetup& iSetup) const {
    if (helixVerbosity_ > 0) {
        std::cout << "DEBUG PZ =====" << std::endl;
        FastHelix helix(inner,middle,outer,iSetup);
        GlobalVector gv=helix.stateAtVertex().parameters().momentum(); // status on inner hit
        std::cout << "FastHelix P = " << gv   << "\n";
        std::cout << "FastHelix Q = " << helix.stateAtVertex().parameters().charge() << "\n";
    }

    // My attempt (with different approx from FastHelix)
    // 1) fit the circle
    FastCircle theCircle(inner,middle,outer);
    double rho = theCircle.rho();

    // 2) Get the PT
    GlobalVector tesla = magfield->inTesla(middle);
    double pt = 0.01 * rho * (0.3*tesla.z());

    // 3) Get the PX,PY at OUTER hit (VERTEX)
    double dx1 = outer.x()-theCircle.x0();
    double dy1 = outer.y()-theCircle.y0();
    double py = pt*dx1/rho, px = -pt*dy1/rho;
    if(px*(middle.x() - outer.x()) + py*(middle.y() - outer.y()) < 0.) {
        px *= -1.; py *= -1.;
    }

    // 4) Get the PZ through pz = pT*(dz/d(R*phi)))
    double dz = inner.z() - outer.z();
    double sinphi = ( dx1*(inner.y()-theCircle.y0()) - dy1*(inner.x()-theCircle.x0())) / (rho * rho);
    double dphi = std::abs(std::asin(sinphi));
    double pz = pt * dz / (dphi * rho); 

    int myq = ((theCircle.x0()*py - theCircle.y0()*px) / tesla.z()) > 0. ?  +1 : -1;
    
    std::pair<GlobalVector,int> mypq(GlobalVector(px,py,pz),myq);

    if (helixVerbosity_ > 1) {
        std::cout << "Gio: pt = " << pt << std::endl;
        std::cout << "Gio: dz = " << dz << ", sinphi = " << sinphi << ", dphi = " << dphi << ", dz/drphi = " << (dz/dphi/rho) << std::endl;
    }
    if (helixVerbosity_ > 0) {
        std::cout << "Gio's fit P = " << mypq.first << "\n";
        std::cout << "Gio's fit Q = " << myq  << "\n";
    }

    return mypq;
}

bool SimpleCosmicBONSeeder::seeds(TrajectorySeedCollection &output, const edm::EventSetup& iSetup)
{
    typedef TrajectoryStateOnSurface TSOS;
    
    for (size_t it=0;it<hitTriplets.size();it++){
        const OrderedHitTriplet &trip = hitTriplets[it];

        GlobalPoint inner = tracker->idToDet((*(trip.inner())).geographicalId())->surface().
            toGlobal((*(trip.inner())).localPosition());

        GlobalPoint middle = tracker->idToDet((*(trip.middle())).geographicalId())->surface().
            toGlobal((*(trip.middle())).localPosition());

        GlobalPoint outer = tracker->idToDet((*(trip.outer())).geographicalId())->surface().
            toGlobal((*(trip.outer())).localPosition());   

        if (seedVerbosity_ > 1)
            std::cout << "Processing triplet " << it << ": " << inner << " + " << middle << " + " << outer << std::endl;

        if ( (outer.y()-inner.y())*outer.y() < 0 ) {
            std::swap(inner,outer);
            std::swap(const_cast<TransientTrackingRecHit::ConstRecHitPointer &>(trip.inner()),
                      const_cast<TransientTrackingRecHit::ConstRecHitPointer &>(trip.outer()) );

//            std::swap(const_cast<ctfseeding::SeedingHit &>(trip.inner()), 
//                      const_cast<ctfseeding::SeedingHit &>(trip.outer()) );
            if (seedVerbosity_ > 1) {
                std::cout << "The seed was going away from CMS! swapped in <-> out" << std::endl;
                std::cout << "Processing swapped triplet " << it << ": " << inner << " + " << middle << " + " << outer << std::endl;
            }
        }

        // First use FastHelix out of the box
        std::pair<GlobalVector,int> pq = pqFromHelixFit(inner,middle,outer,iSetup);
        GlobalVector gv = pq.first;
        float        ch = pq.second; 
        float Mom = sqrt( gv.x()*gv.x() + gv.y()*gv.y() + gv.z()*gv.z() ); 

        if(Mom > 10000 || edm::isNotFinite(Mom))  { 
            if (seedVerbosity_ > 1)
                std::cout << "Processing triplet " << it << ": fail for momentum." << std::endl; 
            continue;
        }

        if (gv.perp() < region_.ptMin()) {
            if (seedVerbosity_ > 1)
                std::cout << "Processing triplet " << it << ": fail for pt = " << gv.perp() << " < ptMin = " << region_.ptMin() << std::endl; 
            continue;
        }

        const Propagator * propagator = 0;  
        if((outer.y()-inner.y())>0){
            if (seedVerbosity_ > 1)
                std::cout << "Processing triplet " << it << ":  downgoing." << std::endl; 
            propagator = thePropagatorAl;
        } else {
            gv = -1*gv; ch = -1.*ch;                        
            propagator = thePropagatorOp;
            if (seedVerbosity_ > 1)
                std::cout << "Processing triplet " << it << ":  upgoing." << std::endl; 
        }

        if (seedVerbosity_ > 1) {
            if (( gv.z() * (outer.z()-inner.z()) > 0 ) && ( fabs(outer.z()-inner.z()) > 5) && (fabs(gv.z()) > .01))  {
                std::cout << "ORRORE: outer.z()-inner.z() = " << (outer.z()-inner.z()) << ", gv.z() = " << gv.z() << std::endl;
            }
        }

        GlobalTrajectoryParameters Gtp(outer,
                gv,int(ch), 
                &(*magfield));
        FreeTrajectoryState CosmicSeed(Gtp,
                CurvilinearTrajectoryError(AlgebraicSymMatrix55(AlgebraicMatrixID())));  
        CosmicSeed.rescaleError(100);
        if (seedVerbosity_ > 2) {
            std::cout << "Processing triplet " << it << ". start from " << std::endl;
            std::cout << "    X  = " << outer << ", P = " << gv << std::endl;
            std::cout << "    Cartesian error (X,P) = \n" << CosmicSeed.cartesianError().matrix() << std::endl;
        }
       
        edm::OwnVector<TrackingRecHit> hits;
        OrderedHitTriplet seedHits(trip.outer(),trip.middle(),trip.inner());
        TSOS propagated, updated;
        bool fail = false;
        for (size_t ih = 0; ih < 3; ++ih) {
            if ((ih == 2) && seedOnMiddle_) {
                if (seedVerbosity_ > 2) 
                    std::cout << "Stopping at middle hit, as requested." << std::endl;
                break;
            }
            if (seedVerbosity_ > 2)
                std::cout << "Processing triplet " << it << ", hit " << ih << "." << std::endl;
            if (ih == 0) {
                propagated = propagator->propagate(CosmicSeed, tracker->idToDet((*seedHits[ih]).geographicalId())->surface());
            } else {
                propagated = propagator->propagate(updated, tracker->idToDet((*seedHits[ih]).geographicalId())->surface());
            }
            if (!propagated.isValid()) {
                if (seedVerbosity_ > 1)
                    std::cout << "Processing triplet " << it << ", hit " << ih << ": failed propagation." << std::endl;
                fail = true; break;
            } else {
                if (seedVerbosity_ > 2)
                    std::cout << "Processing triplet " << it << ", hit " << ih << ": propagated state = " << propagated;
            }
            const TransientTrackingRecHit::ConstRecHitPointer & tthp   = seedHits[ih];
            TransientTrackingRecHit::RecHitPointer              newtth = tthp->clone(propagated);
            hits.push_back(newtth->hit()->clone());
            updated = theUpdator->update(propagated, *newtth);
            if (!updated.isValid()) {
                if (seedVerbosity_ > 1)
                    std::cout << "Processing triplet " << it << ", hit " << ih << ": failed update." << std::endl;
                fail = true; break;
            } else {
                if (seedVerbosity_ > 2)
                    std::cout << "Processing triplet " << it << ", hit " << ih << ": updated state = " << updated;
            }
        }
        if (!fail && updated.isValid() && (updated.globalMomentum().perp() < region_.ptMin())) {
            if (seedVerbosity_ > 1)
                std::cout << "Processing triplet " << it << 
                             ": failed for final pt " << updated.globalMomentum().perp() << " < " << region_.ptMin() << std::endl;
            fail = true;
        }
        if (!fail && updated.isValid() && (updated.globalMomentum().mag() < pMin_)) {
            if (seedVerbosity_ > 1)
                std::cout << "Processing triplet " << it << 
                             ": failed for final p " << updated.globalMomentum().perp() << " < " << pMin_ << std::endl;
            fail = true;
        }
        if (!fail) {
            if (rescaleError_ != 1.0) {
                if (seedVerbosity_ > 2) {
                    std::cout << "Processing triplet " << it << ", rescale error by " << rescaleError_ << ": state BEFORE rescaling " << updated;
                    std::cout << "    Cartesian error (X,P) before rescaling= \n" << updated.cartesianError().matrix() << std::endl;
                }
                updated.rescaleError(rescaleError_);
            }
            if (seedVerbosity_ > 0) {
                std::cout << "Processed  triplet " << it << ": success (saved as #"<<output.size()<<") : " 
                        << inner << " + " << middle << " + " << outer << std::endl;
                std::cout << "    pt = " << updated.globalMomentum().perp() <<
                             "    eta = " << updated.globalMomentum().eta() << 
                             "    phi = " << updated.globalMomentum().phi() <<
                             "    ch = " << updated.charge() << std::endl;
                if (seedVerbosity_ > 1) {
                    std::cout << "    State:" << updated;
                } else {
                    std::cout << "    X  = " << updated.globalPosition() << ", P = " << updated.globalMomentum() << std::endl;
                }
                std::cout << "    Cartesian error (X,P) = \n" << updated.cartesianError().matrix() << std::endl;
            }
            
            PTrajectoryStateOnDet const &  PTraj = trajectoryStateTransform::persistentState(updated, 
                                                            (*(seedOnMiddle_ ? trip.middle() : trip.inner())).geographicalId().rawId());
            output.push_back(TrajectorySeed(PTraj,hits,
                                                ( (outer.y()-inner.y()>0) ? alongMomentum : oppositeToMomentum) ));
            if ((maxSeeds_ > 0) && (output.size() > size_t(maxSeeds_))) { 
                output.clear(); 
                edm::LogError("TooManySeeds") << "Found too many seeds, bailing out.\n";
                return false;
            }
        }
    }
    return true;
}

void SimpleCosmicBONSeeder::done(){
  delete thePropagatorAl;
  delete thePropagatorOp;
  delete theUpdator;
}


