#ifndef Validation_HGCalValidation_CaloParticleSelector_h
#define Validation_HGCalValidation_CaloParticleSelector_h

#include "SimDataFormats/CaloAnalysis/interface/CaloParticle.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/Math/interface/PtEtaPhiMass.h"
#include "DataFormats/Math/interface/deltaPhi.h"

class CaloParticleSelector {
public:
  CaloParticleSelector() {}
  CaloParticleSelector(double ptMin,
                       double ptMax,
                       double minRapidity,
                       double maxRapidity,
                       double lip,
                       double tip,
                       int minHit,
                       unsigned int maxSimClusters,
                       bool signalOnly,
                       bool intimeOnly,
                       bool chargedOnly,
                       bool stableOnly,
                       const std::vector<int>& pdgId = std::vector<int>(),
                       double minPhi = -3.2,
                       double maxPhi = 3.2)
      : ptMin2_(ptMin * ptMin),
        ptMax2_(ptMax * ptMax),
        minRapidity_(minRapidity),
        maxRapidity_(maxRapidity),
        lip_(lip),
        tip2_(tip * tip),
        meanPhi_((minPhi + maxPhi) / 2.),
        rangePhi_((maxPhi - minPhi) / 2.),
        minHit_(minHit),
        maxSimClusters_(maxSimClusters),
        signalOnly_(signalOnly),
        intimeOnly_(intimeOnly),
        chargedOnly_(chargedOnly),
        stableOnly_(stableOnly),
        pdgId_(pdgId) {
    if (minPhi >= maxPhi) {
      throw cms::Exception("Configuration")
          << "CaloParticleSelector: minPhi (" << minPhi << ") must be smaller than maxPhi (" << maxPhi
          << "). The range is constructed from minPhi to maxPhi around their average.";
    }
    if (minPhi >= M_PI) {
      throw cms::Exception("Configuration")
          << "CaloParticleSelector: minPhi (" << minPhi
          << ") must be smaller than PI. The range is constructed from minPhi to maxPhi around their average.";
    }
    if (maxPhi <= -M_PI) {
      throw cms::Exception("Configuration")
          << "CaloParticleSelector: maxPhi (" << maxPhi
          << ") must be larger than -PI. The range is constructed from minPhi to maxPhi around their average.";
    }
  }

  // Operator() performs the selection: e.g. if (cPSelector(cp)) {...}
  // For the moment there shouldn't be any SimTracks from different crossings in the CaloParticle.
  bool operator()(const CaloParticle& cp, std::vector<SimVertex> const& simVertices) const {
    // signal only means no PU particles
    if (signalOnly_ && !(cp.eventId().bunchCrossing() == 0 && cp.eventId().event() == 0))
      return false;
    // intime only means no OOT PU particles
    if (intimeOnly_ && !(cp.eventId().bunchCrossing() == 0))
      return false;

    if (cp.simClusters().size() > maxSimClusters_)
      return false;

    auto pdgid = cp.pdgId();
    if (!pdgId_.empty()) {
      bool testId = false;
      for (auto id : pdgId_) {
        if (id == pdgid) {
          testId = true;
          break;
        }
      }
      if (!testId)
        return false;
    }

    if (chargedOnly_ && cp.charge() == 0)
      return false;  //select only if charge!=0

    // select only stable particles
    if (stableOnly_) {
      for (CaloParticle::genp_iterator j = cp.genParticle_begin(); j != cp.genParticle_end(); ++j) {
        if (j->get() == nullptr || j->get()->status() != 1) {
          return false;
        }
      }

      // test for remaining unstabled due to lack of genparticle pointer
      std::vector<int> pdgids{11, 13, 211, 321, 2212, 3112, 3222, 3312, 3334};
      if (cp.status() == -99 && (!std::binary_search(pdgids.begin(), pdgids.end(), std::abs(pdgid)))) {
        return false;
      }
    }

    auto etaOk = [&](const CaloParticle& p) -> bool {
      float eta = etaFromXYZ(p.px(), p.py(), p.pz());
      return (eta >= minRapidity_) & (eta <= maxRapidity_);
    };
    auto phiOk = [&](const CaloParticle& p) {
      float dphi = deltaPhi(atan2f(p.py(), p.px()), meanPhi_);
      return dphi >= -rangePhi_ && dphi <= rangePhi_;
    };
    auto ptOk = [&](const CaloParticle& p) {
      double pt2 = cp.p4().perp2();
      return pt2 >= ptMin2_ && pt2 <= ptMax2_;
    };

    return (ptOk(cp) && etaOk(cp) && phiOk(cp));
  }

private:
  double ptMin2_;
  double ptMax2_;
  float minRapidity_;
  float maxRapidity_;
  double lip_;
  double tip2_;
  float meanPhi_;
  float rangePhi_;
  int minHit_;
  unsigned int maxSimClusters_;
  bool signalOnly_;
  bool intimeOnly_;
  bool chargedOnly_;
  bool stableOnly_;
  std::vector<int> pdgId_;
};

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "CommonTools/UtilAlgos/interface/ParameterAdapter.h"

namespace reco {
  namespace modules {

    template <>
    struct ParameterAdapter<CaloParticleSelector> {
      static CaloParticleSelector make(const edm::ParameterSet& cfg, edm::ConsumesCollector& iC) { return make(cfg); }

      static CaloParticleSelector make(const edm::ParameterSet& cfg) {
        return CaloParticleSelector(cfg.getParameter<double>("ptMinCP"),
                                    cfg.getParameter<double>("ptMaxCP"),
                                    cfg.getParameter<double>("minRapidityCP"),
                                    cfg.getParameter<double>("maxRapidityCP"),
                                    cfg.getParameter<double>("lip"),
                                    cfg.getParameter<double>("tip"),
                                    cfg.getParameter<int>("minHitCP"),
                                    cfg.getParameter<int>("maxSimClustersCP"),
                                    cfg.getParameter<bool>("signalOnlyCP"),
                                    cfg.getParameter<bool>("intimeOnlyCP"),
                                    cfg.getParameter<bool>("chargedOnlyCP"),
                                    cfg.getParameter<bool>("stableOnlyCP"),
                                    cfg.getParameter<std::vector<int> >("pdgIdCP"),
                                    cfg.getParameter<double>("minPhiCP"),
                                    cfg.getParameter<double>("maxPhiCP"));
      }
    };

  }  // namespace modules
}  // namespace reco

#endif
