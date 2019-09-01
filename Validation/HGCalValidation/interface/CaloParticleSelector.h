#ifndef Validation_HGCalValidation_CaloParticleSelector_h
#define Validation_HGCalValidation_CaloParticleSelector_h
/* \class CaloParticleSelector
 *
 * \author Giuseppe Cerati, INFN
 *
 *  $Date: 2013/05/14 15:46:46 $
 *  $Revision: 1.5.4.2 $
 *
 */
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
  bool operator()(const CaloParticle& tp, std::vector<SimVertex> const& simVertices) const {
    // signal only means no PU particles
    if (signalOnly_ && !(tp.eventId().bunchCrossing() == 0 && tp.eventId().event() == 0))
      return false;
    // intime only means no OOT PU particles
    if (intimeOnly_ && !(tp.eventId().bunchCrossing() == 0))
      return false;

    auto pdgid = tp.pdgId();
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

    if (chargedOnly_ && tp.charge() == 0)
      return false;  //select only if charge!=0

    // select only stable particles
    if (stableOnly_) {
      for (CaloParticle::genp_iterator j = tp.genParticle_begin(); j != tp.genParticle_end(); ++j) {
        if (j->get() == nullptr || j->get()->status() != 1) {
          return false;
        }
      }

      // test for remaining unstabled due to lack of genparticle pointer
      std::vector<int> pdgids{11, 13, 211, 321, 2212, 3112, 3222, 3312, 3334};
      if (tp.status() == -99 && (!std::binary_search(pdgids.begin(), pdgids.end(), std::abs(pdgid)))) {
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
      double pt2 = tp.p4().perp2();
      return pt2 >= ptMin2_ && pt2 <= ptMax2_;
    };

    return (ptOk(tp) && etaOk(tp) && phiOk(tp));
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
        return CaloParticleSelector(cfg.getParameter<double>("ptMin"),
                                    cfg.getParameter<double>("ptMax"),
                                    cfg.getParameter<double>("minRapidity"),
                                    cfg.getParameter<double>("maxRapidity"),
                                    cfg.getParameter<double>("lip"),
                                    cfg.getParameter<double>("tip"),
                                    cfg.getParameter<int>("minHit"),
                                    cfg.getParameter<bool>("signalOnly"),
                                    cfg.getParameter<bool>("intimeOnly"),
                                    cfg.getParameter<bool>("chargedOnly"),
                                    cfg.getParameter<bool>("stableOnly"),
                                    cfg.getParameter<std::vector<int> >("pdgId"),
                                    cfg.getParameter<double>("minPhi"),
                                    cfg.getParameter<double>("maxPhi"));
      }
    };

  }  // namespace modules
}  // namespace reco

#endif
