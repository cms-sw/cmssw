//
// Package:         RecoTracker/TkSeedGenerator
// Class:           GlobalPixelLessSeedGenerator
// 
// From CosmicSeedGenerator+SimpleCosmicBONSeeder, with changes by Giovanni
// to seed Cosmics with B != 0

#include "RecoTracker/SpecialSeedGenerators/interface/SimpleCosmicBONSeeder.h"
#include "RecoTracker/TkSeedGenerator/interface/FastLine.h"

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
  check_(conf.getParameter<edm::ParameterSet>("ClusterCheckPSet"))
{
  edm::ParameterSet regionConf = conf_.getParameter<edm::ParameterSet>("RegionPSet");
  float ptmin        = regionConf.getParameter<double>("ptMin");
  float originradius = regionConf.getParameter<double>("originRadius");
  float halflength   = regionConf.getParameter<double>("originHalfLength");
  float originz      = regionConf.getParameter<double>("originZPosition");
  region = GlobalTrackingRegion(ptmin, originradius, halflength, originz);

  builderName = conf_.getParameter<std::string>("TTRHBuilder");   

  produces<TrajectorySeedCollection>();
  if (writeTriplets_) produces<edm::OwnVector<TrackingRecHit> >("cosmicTriplets");
}

// Functions that gets called by framework every event
void SimpleCosmicBONSeeder::produce(edm::Event& ev, const edm::EventSetup& es)
{
  std::auto_ptr<TrajectorySeedCollection> output(new TrajectorySeedCollection());
  std::auto_ptr<edm::OwnVector<TrackingRecHit> > outtriplets(new edm::OwnVector<TrackingRecHit>());

  es.get<IdealMagneticFieldRecord>().get(magfield);
  if ((magfield->inTesla(GlobalPoint(0,0,0)).mag() > 0.01) && !check_.tooManyClusters(ev)){
    init(es);
    triplets(ev,es);
    seedsOutIn(*output,es);

    if (writeTriplets_) {
        for (OrderedHitTriplets::const_iterator it = hitTriplets.begin(); it != hitTriplets.end(); ++it) {
            const TrackingRecHit * hit1 = it->inner();    
            const TrackingRecHit * hit2 = it->middle();
            const TrackingRecHit * hit3 = it->outer();
            outtriplets->push_back(hit1->clone());
            outtriplets->push_back(hit2->clone());
            outtriplets->push_back(hit3->clone());
        }
    }

    done();
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
                // sort by inner, or by outer
                return (iy1 < iy2 ? true : (oy1 < oy2));
            } else return true; // else prefer downgoing
        } else if (oy2 - iy2 > 0) {
            return true; // prefer downgoing
        } else {
            // sort by inner, or by outer
            return (iy1 < iy2 ? true : (oy1 < oy2));
        }
    }
};

void SimpleCosmicBONSeeder::triplets(const edm::Event& e, const edm::EventSetup& es) {
    using namespace ctfseeding;

    hitTriplets.clear();
    hitTriplets.reserve(0);
    SeedingLayerSets lss = theLsb.layers(es);
    SeedingLayerSets::const_iterator iLss;
    for (iLss = lss.begin(); iLss != lss.end(); iLss++){
        SeedingLayers ls = *iLss;
        if (ls.size() != 3){
            throw cms::Exception("CtfSpecialSeedGenerator") << "You are using " << ls.size() <<" layers in set instead of 3 ";
        }
        std::vector<SeedingHit> innerHits  = region.hits(e, es, &ls[0]);
        std::vector<SeedingHit> middleHits = region.hits(e, es, &ls[1]);
        std::vector<SeedingHit> outerHits  = region.hits(e, es, &ls[2]);

        if (tripletsVerbosity_ > 0) std::cout << "GenericTripletGenerator iLss = (" << (iLss - lss.begin()) << "): # = " <<
            innerHits.size() << "/" << middleHits.size() << "/" << outerHits.size() << std::endl;

        std::vector<SeedingHit>::const_iterator iOuterHit,iMiddleHit,iInnerHit;
        for (iOuterHit = outerHits.begin(); iOuterHit != outerHits.end(); iOuterHit++){
            for (iMiddleHit = middleHits.begin(); iMiddleHit != middleHits.end(); iMiddleHit++){
                for (iInnerHit = innerHits.begin(); iInnerHit != innerHits.end(); iInnerHit++){
                    GlobalPoint innerpos  = ls[0].hitBuilder()->build(&(**iInnerHit))->globalPosition();
                    GlobalPoint middlepos = ls[1].hitBuilder()->build(&(**iMiddleHit))->globalPosition();
                    GlobalPoint outerpos  = ls[2].hitBuilder()->build(&(**iOuterHit))->globalPosition();
                    if (tripletsVerbosity_ > 1) std::cout << "Trying seed with: " << innerpos << " + " << middlepos << " + " << outerpos << std::endl;
                    if (goodTriplet(innerpos,middlepos,outerpos)) {
                        OrderedHitTriplet oht(*iInnerHit,*iMiddleHit,*iOuterHit);
                        hitTriplets.push_back(oht);
                        if (tripletsVerbosity_ > 2)  std::cout << " accepted seed w/: " << innerpos << " + " << middlepos << " + " << outerpos << std::endl;
                        if (tripletsVerbosity_ == 1) std::cout << " good seed w/: "     << innerpos << " + " << middlepos << " + " << outerpos << std::endl;
                        if (tripletsVerbosity_ > 2 && (helixVerbosity_ > 0)) { // debug the momentum here too
                            pqFromHelixFit(innerpos,middlepos,outerpos,es); 
                        }
                    }
                }
            }
        }
    }
    //std::sort(hitTriplets.begin(),hitTriplets.end(),HigherInnerHit());
}

bool SimpleCosmicBONSeeder::goodTriplet(const GlobalPoint &inner, const GlobalPoint & middle, const GlobalPoint & outer) const {
    float dyOM = outer.y() - middle.y(), dyIM = inner.y() - middle.y();
    if ((dyOM * dyIM > 0) && (fabs(dyOM)>10) && (fabs(dyIM)>10)) {
        if (tripletsVerbosity_ > 1) std::cout << "  fail for non coherent dy" << std::endl;
        return false;
    }
    float dzOM = outer.z() - middle.z(), dzIM = inner.z() - middle.z();
    if ((dzOM * dzIM > 0) && (fabs(dzOM)>50) && (fabs(dzIM)>50)) {
        if (tripletsVerbosity_ > 1) std::cout << "  fail for non coherent dz" << std::endl;
        return false;
    }
    return true;
}

void SimpleCosmicBONSeeder::seedsInOut(TrajectorySeedCollection &output, const edm::EventSetup& iSetup)
{
#if 0
    typedef TrajectoryStateOnSurface TSOS;

    for (size_t it=0;it<hitTriplets.size();it++){
        const OrderedHitTriplet &trip = hitTriplets[it];

        GlobalPoint inner = tracker->idToDet((*(trip.inner())).geographicalId())->surface().
            toGlobal((*(trip.inner())).localPosition());

        GlobalPoint middle = tracker->idToDet((*(trip.middle())).geographicalId())->surface().
            toGlobal((*(trip.middle())).localPosition());

        GlobalPoint outer = tracker->idToDet((*(trip.outer())).geographicalId())->surface().
            toGlobal((*(trip.outer())).localPosition());   

        std::cout << "Processing triplet " << it << ": " << inner << " + " << middle << " + " << outer << std::endl;


        FastHelix helix(outer,middle,inner,iSetup);
        GlobalVector gv=helix.stateAtVertex().parameters().momentum(); // status on inner hit
        float ch=helix.stateAtVertex().parameters().charge();
        float Mom = sqrt( gv.x()*gv.x() + gv.y()*gv.y() + gv.z()*gv.z() ); 

        if(Mom > 1000 || isnan(Mom))  { 
            std::cout << "Processing triplet " << it << ": fail for momentum." << std::endl; 
            continue;
        }

        if (gv.y()<0){ 
            gv=-1.*gv; ch=-1.*ch; 
            std::cout << "Processing triplet " << it << ": flip charge." << std::endl; 
        }

        GlobalTrajectoryParameters Gtp(inner,
                gv,int(ch), 
                &(*magfield));
        FreeTrajectoryState CosmicSeed(Gtp,
                CurvilinearTrajectoryError(AlgebraicSymMatrix55(AlgebraicMatrixID())));  
        CosmicSeed.rescaleError(100);
       
        const Propagator * propagator = 0;  
        if((outer.y()-inner.y())>0){
            std::cout << "Processing triplet " << it << ":  downgoing." << std::endl; 
            propagator = thePropagatorOp;
        } else {
            std::cout << "Processing triplet " << it << ":  upgoing." << std::endl; 
            propagator = thePropagatorAl;
        }
        const OrderedHitTriplet::Hits seedHits = trip.hits(); // in,mid,out
        TSOS propagated, updated;
        bool fail = false;
        for (size_t ih = 0; ih < 3; ++ih) {
            //std::cout << "Processing triplet " << it << ", hit " << ih << "." << std::endl;
            if (ih == 0) {
                propagated = propagator->propagate(CosmicSeed, tracker->idToDet((*seedHits[ih]).geographicalId())->surface());
            } else {
                propagated = propagator->propagate(updated, tracker->idToDet((*seedHits[ih]).geographicalId())->surface());
            }
            if (!propagated.isValid()) {
                std::cout << "Processing triplet " << it << ", hit " << ih << ": failed propagation." << std::endl;
                fail = true; break;
            } else {
                //std::cout << "Processing triplet " << it << ", hit " << ih << ": propagated state = " << propagated;
            }
            const TransientTrackingRecHit::ConstRecHitPointer & tthp   = seedHits[ih];
            TransientTrackingRecHit::RecHitPointer              newtth = tthp->clone(propagated);
            updated = theUpdator->update(propagated, *newtth);
            if (!updated.isValid()) {
                std::cout << "Processing triplet " << it << ", hit " << ih << ": failed update." << std::endl;
                fail = true; break;
            } else {
                //std::cout << "Processing triplet " << it << ", hit " << ih << ": updated state = " << updated;
            }
        }

        if (!fail) {
            std::cout << "Processing triplet " << it << ": finally made a state:" << updated <<"\n Now flipping it." << std::endl;
            TSOS flipped( LocalTrajectoryParameters(updated.localPosition(), - updated.localMomentum(), updated.charge()),
                          updated.localError(), updated.surface(), updated.magneticField());
            flipped.rescaleError(10);
            std::cout << "Processing triplet " << it << ": flipped and rescaled state is " << flipped << std::endl;
            std::cout << "Processed  triplet " << it << ": success: " << inner << " + " << middle << " + " << outer << std::endl;
            
            PTrajectoryStateOnDet *PTraj = transformer.persistentState(flipped, (*(trip.outer())).geographicalId().rawId());
            edm::OwnVector<TrackingRecHit> hits;
            output.push_back(TrajectorySeed(*PTraj,hits,
                                                ( (outer.y()-inner.y()>0) ? alongMomentum : oppositeToMomentum) ));
            delete PTraj;
        }
    }
#endif
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
    double dphi = std::asin(sinphi);
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

void SimpleCosmicBONSeeder::seedsOutIn(TrajectorySeedCollection &output, const edm::EventSetup& iSetup)
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

        // First use FastHelix out of the box
        std::pair<GlobalVector,int> pq = pqFromHelixFit(inner,middle,outer,iSetup);
        GlobalVector gv = pq.first;
        float        ch = pq.second; 
        float Mom = sqrt( gv.x()*gv.x() + gv.y()*gv.y() + gv.z()*gv.z() ); 

        if(Mom > 1000 || isnan(Mom))  { 
            if (seedVerbosity_ > 1)
                std::cout << "Processing triplet " << it << ": fail for momentum." << std::endl; 
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
        OrderedHitTriplet::Hits seedHits;
        seedHits.push_back(trip.outer()); 
        seedHits.push_back(trip.middle()); 
        seedHits.push_back(trip.inner()); 
        TSOS propagated, updated;
        bool fail = false;
        for (size_t ih = 0; ih < 3; ++ih) {
            if ((ih == 2) && seedOnMiddle_) {
                if (seedVerbosity_ > 2) 
                    std::cout << "Stopping at middle hit, as requested." << std::endl;
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

        if (!fail) {
            if (rescaleError_ != 1.0) {
                if (seedVerbosity_ > 2) {
                    std::cout << "Processing triplet " << it << ", rescale error by " << rescaleError_ << ": state BEFORE rescaling " << updated;
                    std::cout << "    Cartesian error (X,P) before rescaling= \n" << updated.cartesianError().matrix() << std::endl;
                }
                updated.rescaleError(rescaleError_);
            }
            if (seedVerbosity_ > 0) {
                std::cout << "Processed  triplet " << it << ": success: " << inner << " + " << middle << " + " << outer << std::endl;
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
            
            PTrajectoryStateOnDet *PTraj = transformer.persistentState(updated, (*(trip.inner())).geographicalId().rawId());
            output.push_back(TrajectorySeed(*PTraj,hits,
                                                ( (outer.y()-inner.y()>0) ? alongMomentum : oppositeToMomentum) ));
            delete PTraj;
        }
    }
}

void SimpleCosmicBONSeeder::seeds(TrajectorySeedCollection &output, const edm::EventSetup& iSetup)
{
#if 0
    typedef TrajectoryStateOnSurface TSOS;

    for (uint it=0;it<hitTriplets.size();it++){

        GlobalPoint inner = tracker->idToDet((*(hitTriplets[it].inner())).geographicalId())->surface().
            toGlobal((*(hitTriplets[it].inner())).localPosition());

        GlobalPoint middle = tracker->idToDet((*(hitTriplets[it].middle())).geographicalId())->surface().
            toGlobal((*(hitTriplets[it].middle())).localPosition());

        GlobalPoint outer = tracker->idToDet((*(hitTriplets[it].outer())).geographicalId())->surface().
            toGlobal((*(hitTriplets[it].outer())).localPosition());   

        TransientTrackingRecHit::ConstRecHitPointer outrhit= TTTRHBuilder->build(hitTriplets[it].outer());
        edm::OwnVector<TrackingRecHit> hits;
        hits.push_back((*(hitTriplets[it].outer())).clone());
        FastHelix helix(inner, middle, outer,iSetup);
        GlobalVector gv=helix.stateAtVertex().parameters().momentum();
        float ch=helix.stateAtVertex().parameters().charge();
        float Mom = sqrt( gv.x()*gv.x() + gv.y()*gv.y() + gv.z()*gv.z() ); 
        if(Mom > 1000 || isnan(Mom))  continue;   // ChangedByDaniele 

        if (gv.y()>0){
            gv=-1.*gv;
            ch=-1.*ch;
        }

        GlobalTrajectoryParameters Gtp(outer,
                gv,int(ch), 
                &(*magfield));
        FreeTrajectoryState CosmicSeed(Gtp,
                CurvilinearTrajectoryError(AlgebraicSymMatrix55(AlgebraicMatrixID())));  
        if((outer.y()-inner.y())>0){
            const TSOS outerState =
                thePropagatorAl->propagate(CosmicSeed,
                        tracker->idToDet((*(hitTriplets[it].outer())).geographicalId())->surface());
            if ( outerState.isValid()) {
                LogDebug("SimpleCosmicBONSeeder") <<"outerState "<<outerState;
                const TSOS outerUpdated= theUpdator->update( outerState,*outrhit);
                if ( outerUpdated.isValid()) {
                    LogDebug("SimpleCosmicBONSeeder") <<"outerUpdated "<<outerUpdated;

                    PTrajectoryStateOnDet *PTraj=  
                        transformer.persistentState(outerUpdated,(*(hitTriplets[it].outer())).geographicalId().rawId());
                    output.push_back(TrajectorySeed(*PTraj,hits,alongMomentum));

                    delete PTraj;
                }
            }
        } else {
            const TSOS outerState =
                thePropagatorOp->propagate(CosmicSeed,
                        tracker->idToDet((*(hitTriplets[it].outer())).geographicalId())->surface());
            if ( outerState.isValid()) {
                LogDebug("SimpleCosmicBONSeeder") <<"outerState "<<outerState;
                const TSOS outerUpdated= theUpdator->update( outerState,*outrhit);
                if ( outerUpdated.isValid()) {
                    LogDebug("SimpleCosmicBONSeeder") <<"outerUpdated "<<outerUpdated;

                    PTrajectoryStateOnDet *PTraj=  
                        transformer.persistentState(outerUpdated, (*(hitTriplets[it].outer())).geographicalId().rawId());
                    output.push_back(TrajectorySeed(*PTraj,hits,oppositeToMomentum));
                    delete PTraj;
                }
            }
        }
    }
#endif
}
void SimpleCosmicBONSeeder::done(){
  delete thePropagatorAl;
  delete thePropagatorOp;
  delete theUpdator;
}


