#include "RecoTracker/TkSeedingLayers/interface/SeedComparitor.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedComparitorFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <cmath>

class SimpleClusterProbabilitySeedComparitor : public SeedComparitor {
    public:
        SimpleClusterProbabilitySeedComparitor(const edm::ParameterSet &cfg) ;
        virtual ~SimpleClusterProbabilitySeedComparitor() ; 
        virtual void init(const edm::EventSetup& es) {}
        virtual bool compatible(const SeedingHitSet  &hits, const TrackingRegion & region) { return true; }
        virtual bool compatible(const TrajectorySeed &seed) const { return true; }
        virtual bool compatible(const TrajectoryStateOnSurface &,
                const TransientTrackingRecHit::ConstRecHitPointer &hit) const ;
        virtual bool compatible(const SeedingHitSet  &hits, 
                const GlobalTrajectoryParameters &helixStateAtVertex,
                const FastHelix                  &helix,
                const TrackingRegion & region) const { return true; }
        virtual bool compatible(const SeedingHitSet  &hits, 
                const GlobalTrajectoryParameters &straightLineStateAtVertex,
                const TrackingRegion & region) const { return true; }


    private:
        float probCut_;
};


SimpleClusterProbabilitySeedComparitor::SimpleClusterProbabilitySeedComparitor(const edm::ParameterSet &cfg) :
    probCut_(cfg.getParameter<double>("LogPixelProbabilityCut"))
{
}

SimpleClusterProbabilitySeedComparitor::~SimpleClusterProbabilitySeedComparitor() 
{
}

bool
SimpleClusterProbabilitySeedComparitor::compatible(const TrajectoryStateOnSurface &tsos,
                const TransientTrackingRecHit::ConstRecHitPointer &hit) const
{
    return (probCut_ < -15.) || (log10(hit->clusterProbability()) > probCut_);
}

DEFINE_EDM_PLUGIN(SeedComparitorFactory, SimpleClusterProbabilitySeedComparitor, "SimpleClusterProbabilitySeedComparitor");
