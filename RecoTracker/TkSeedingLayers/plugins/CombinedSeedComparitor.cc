#include "RecoTracker/TkSeedingLayers/interface/SeedComparitor.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedComparitorFactory.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <boost/ptr_container/ptr_vector.hpp>

class CombinedSeedComparitor : public SeedComparitor {
public:
  CombinedSeedComparitor(const edm::ParameterSet &cfg, edm::ConsumesCollector &iC);
  ~CombinedSeedComparitor() override;
  void init(const edm::Event &ev, const edm::EventSetup &es) override;
  bool compatible(const SeedingHitSet &hits) const override;
  bool compatible(const TrajectoryStateOnSurface &, SeedingHitSet::ConstRecHitPointer hit) const override;
  bool compatible(const SeedingHitSet &hits,
                  const GlobalTrajectoryParameters &helixStateAtVertex,
                  const FastHelix &helix) const override;

private:
  boost::ptr_vector<SeedComparitor> comparitors_;
  bool isAnd_;
};

CombinedSeedComparitor::CombinedSeedComparitor(const edm::ParameterSet &cfg, edm::ConsumesCollector &iC) {
  std::string mode = cfg.getParameter<std::string>("mode");
  if (mode == "and")
    isAnd_ = true;
  else if (mode == "or")
    isAnd_ = false;
  else
    throw cms::Exception("Configuration", "Parameter 'mode' of CombinedSeedComparitor must be either 'and' or 'or'\n");

  typedef std::vector<edm::ParameterSet> VPSet;
  VPSet psets = cfg.getParameter<VPSet>("comparitors");
  for (const auto &pset : psets) {
    std::string name = pset.getParameter<std::string>("ComponentName");
    comparitors_.push_back(SeedComparitorFactory::get()->create(name, pset, iC));
  }
}

CombinedSeedComparitor::~CombinedSeedComparitor() {}

void CombinedSeedComparitor::init(const edm::Event &ev, const edm::EventSetup &es) {
  typedef boost::ptr_vector<SeedComparitor>::iterator ITC;
  for (auto &comparitor : comparitors_) {
    comparitor.init(ev, es);
  }
}

bool CombinedSeedComparitor::compatible(const SeedingHitSet &hits) const {
  typedef boost::ptr_vector<SeedComparitor>::const_iterator ITC;
  for (const auto &comparitor : comparitors_) {
    bool pass = comparitor.compatible(hits);
    if (isAnd_ != pass)
      return pass;  // break on failures if doing an AND, and on successes if doing an OR
  }
  return isAnd_;  // if we arrive here, we have no successes for OR, and no failures for AND
}

bool CombinedSeedComparitor::compatible(const TrajectoryStateOnSurface &tsos,
                                        SeedingHitSet::ConstRecHitPointer hit) const {
  typedef boost::ptr_vector<SeedComparitor>::const_iterator ITC;
  for (const auto &comparitor : comparitors_) {
    bool pass = comparitor.compatible(tsos, hit);
    if (isAnd_ != pass)
      return pass;  // break on failures if doing an AND, and on successes if doing an OR
  }
  return isAnd_;  // if we arrive here, we have no successes for OR, and no failures for AND
}

bool CombinedSeedComparitor::compatible(const SeedingHitSet &hits,
                                        const GlobalTrajectoryParameters &helixStateAtVertex,
                                        const FastHelix &helix) const {
  typedef boost::ptr_vector<SeedComparitor>::const_iterator ITC;
  for (const auto &comparitor : comparitors_) {
    bool pass = comparitor.compatible(hits, helixStateAtVertex, helix);
    if (isAnd_ != pass)
      return pass;  // break on failures if doing an AND, and on successes if doing an OR
  }
  return isAnd_;  // if we arrive here, we have no successes for OR, and no failures for AND
}

DEFINE_EDM_PLUGIN(SeedComparitorFactory, CombinedSeedComparitor, "CombinedSeedComparitor");
