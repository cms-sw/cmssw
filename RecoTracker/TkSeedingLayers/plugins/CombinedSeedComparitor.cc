#include "RecoTracker/TkSeedingLayers/interface/SeedComparitor.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedComparitorFactory.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <vector>
#include <memory>

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
  std::vector<std::unique_ptr<SeedComparitor>> comparitors_;
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
  for (VPSet::const_iterator it = psets.begin(), ed = psets.end(); it != ed; ++it) {
    std::string name = it->getParameter<std::string>("ComponentName");
    comparitors_.emplace_back(SeedComparitorFactory::get()->create(name, *it, iC));
  }
}

CombinedSeedComparitor::~CombinedSeedComparitor() {}

void CombinedSeedComparitor::init(const edm::Event &ev, const edm::EventSetup &es) {
  for (const auto &it : comparitors_) {
    it->init(ev, es);
  }
}

bool CombinedSeedComparitor::compatible(const SeedingHitSet &hits) const {
  for (const auto &it : comparitors_) {
    bool pass = it->compatible(hits);
    if (isAnd_ != pass)
      return pass;  // break on failures if doing an AND, and on successes if doing an OR
  }
  return isAnd_;  // if we arrive here, we have no successes for OR, and no failures for AND
}

bool CombinedSeedComparitor::compatible(const TrajectoryStateOnSurface &tsos,
                                        SeedingHitSet::ConstRecHitPointer hit) const {
  for (const auto &it : comparitors_) {
    bool pass = it->compatible(tsos, hit);
    if (isAnd_ != pass)
      return pass;  // break on failures if doing an AND, and on successes if doing an OR
  }
  return isAnd_;  // if we arrive here, we have no successes for OR, and no failures for AND
}

bool CombinedSeedComparitor::compatible(const SeedingHitSet &hits,
                                        const GlobalTrajectoryParameters &helixStateAtVertex,
                                        const FastHelix &helix) const {
  for (const auto &it : comparitors_) {
    bool pass = it->compatible(hits, helixStateAtVertex, helix);
    if (isAnd_ != pass)
      return pass;  // break on failures if doing an AND, and on successes if doing an OR
  }
  return isAnd_;  // if we arrive here, we have no successes for OR, and no failures for AND
}

DEFINE_EDM_PLUGIN(SeedComparitorFactory, CombinedSeedComparitor, "CombinedSeedComparitor");
