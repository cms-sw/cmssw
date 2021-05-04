#ifndef SimTracker_Common_TrackingParticleSelector_h
#define SimTracker_Common_TrackingParticleSelector_h
/* \class TrackingParticleSelector
 *
 * \author Giuseppe Cerati, INFN
 *
 *  $Date: 2013/05/14 15:46:46 $
 *  $Revision: 1.5.4.2 $
 *
 */
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/Math/interface/PtEtaPhiMass.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"

class TrackingParticleSelector {
public:
  TrackingParticleSelector() {}
  TrackingParticleSelector(double ptMin,
                           double ptMax,
                           double minRapidity,
                           double maxRapidity,
                           double tip,
                           double lip,
                           int minHit,
                           bool signalOnly,
                           bool intimeOnly,
                           bool chargedOnly,
                           bool stableOnly,
                           const std::vector<int> &pdgId = std::vector<int>(),
                           bool invertRapidityCut = false,
                           double minPhi = -3.2,
                           double maxPhi = 3.2)
      : ptMin2_(ptMin * ptMin),
        ptMax2_(ptMax * ptMax),
        minRapidity_(minRapidity),
        maxRapidity_(maxRapidity),
        meanPhi_((minPhi + maxPhi) / 2.),
        rangePhi_((maxPhi - minPhi) / 2.),
        tip2_(tip * tip),
        lip_(lip),
        minHit_(minHit),
        signalOnly_(signalOnly),
        intimeOnly_(intimeOnly),
        chargedOnly_(chargedOnly),
        stableOnly_(stableOnly),
        pdgId_(pdgId),
        invertRapidityCut_(invertRapidityCut) {
    if (minPhi >= maxPhi) {
      throw cms::Exception("Configuration")
          << "TrackingParticleSelector: minPhi (" << minPhi << ") must be smaller than maxPhi (" << maxPhi
          << "). The range is constructed from minPhi to maxPhi around their "
             "average.";
    }
    if (minPhi >= M_PI) {
      throw cms::Exception("Configuration") << "TrackingParticleSelector: minPhi (" << minPhi
                                            << ") must be smaller than PI. The range is constructed from minPhi "
                                               "to maxPhi around their average.";
    }
    if (maxPhi <= -M_PI) {
      throw cms::Exception("Configuration") << "TrackingParticleSelector: maxPhi (" << maxPhi
                                            << ") must be larger than -PI. The range is constructed from minPhi "
                                               "to maxPhi around their average.";
    }
  }

  bool isCharged(const TrackingParticle *tp) const { return (tp->charge() == 0 ? false : true); }

  bool isInTime(const TrackingParticle *tp) const { return (tp->eventId().bunchCrossing() == 0); }

  bool isSignal(const TrackingParticle *tp) const {
    return (tp->eventId().bunchCrossing() == 0 && tp->eventId().event() == 0);
  }

  bool isStable(const TrackingParticle *tp) const {
    for (TrackingParticle::genp_iterator j = tp->genParticle_begin(); j != tp->genParticle_end(); ++j) {
      if (j->get() == nullptr || j->get()->status() != 1) {
        return false;
      }
    }
    // test for remaining unstabled due to lack of genparticle pointer
    auto pdgid = tp->pdgId();
    if (tp->status() == -99 && (std::abs(pdgid) != 11 && std::abs(pdgid) != 13 && std::abs(pdgid) != 211 &&
                                std::abs(pdgid) != 321 && std::abs(pdgid) != 2212 && std::abs(pdgid) != 3112 &&
                                std::abs(pdgid) != 3222 && std::abs(pdgid) != 3312 && std::abs(pdgid) != 3334)) {
      return false;
    }
    return true;
  }

  /// Operator() performs the selection: e.g. if (tPSelector(tp)) {...}
  /// https://stackoverflow.com/questions/14466620/c-template-specialization-calling-methods-on-types-that-could-be-pointers-or/14466705
  bool operator()(const TrackingParticle &tp) const { return select(&tp); }
  bool operator()(const TrackingParticle *tp) const { return select(tp); }

  bool select(const TrackingParticle *tp) const {
    // signal only means no PU particles
    if (signalOnly_ && !isSignal(tp))
      return false;
    // intime only means no OOT PU particles
    if (intimeOnly_ && !isInTime(tp))
      return false;

    // select only if charge!=0
    if (chargedOnly_ && !isCharged(tp))
      return false;

    // select for particle type
    if (!selectParticleType(tp)) {
      return false;
    }

    // select only stable particles
    if (stableOnly_ && !isStable(tp)) {
      return false;
    }

    return selectKinematics(tp);
  }

  bool selectKinematics(const TrackingParticle *tp) const {
    auto etaOk = [&](const TrackingParticle *p) -> bool {
      float eta = etaFromXYZ(p->px(), p->py(), p->pz());
      if (!invertRapidityCut_)
        return (eta >= minRapidity_) && (eta <= maxRapidity_);
      else
        return (eta < minRapidity_ || eta > maxRapidity_);
    };
    auto phiOk = [&](const TrackingParticle *p) {
      float dphi = deltaPhi(atan2f(p->py(), p->px()), meanPhi_);
      return dphi >= -rangePhi_ && dphi <= rangePhi_;
    };
    auto ptOk = [&](const TrackingParticle *p) {
      double pt2 = tp->p4().perp2();
      return pt2 >= ptMin2_ && pt2 <= ptMax2_;
    };
    return (tp->numberOfTrackerLayers() >= minHit_ && ptOk(tp) && etaOk(tp) && phiOk(tp) &&
            std::abs(tp->vertex().z()) <= lip_ &&  // vertex last to avoid to load it if not striclty
                                                   // necessary...
            tp->vertex().perp2() <= tip2_);
  }

  bool selectParticleType(const TrackingParticle *tp) const {
    auto pdgid = tp->pdgId();
    if (!pdgId_.empty()) {
      for (auto id : pdgId_) {
        if (id == pdgid) {
          return true;
        }
      }
    } else {
      return true;
    }
    return false;
  }

private:
  double ptMin2_;
  double ptMax2_;
  float minRapidity_;
  float maxRapidity_;
  float meanPhi_;
  float rangePhi_;
  double tip2_;
  double lip_;
  int minHit_;
  bool signalOnly_;
  bool intimeOnly_;
  bool chargedOnly_;
  bool stableOnly_;
  std::vector<int> pdgId_;
  bool invertRapidityCut_;
};

#include "CommonTools/UtilAlgos/interface/ParameterAdapter.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

namespace reco {
  namespace modules {

    template <>
    struct ParameterAdapter<TrackingParticleSelector> {
      static TrackingParticleSelector make(const edm::ParameterSet &cfg, edm::ConsumesCollector &iC) {
        return make(cfg);
      }

      static TrackingParticleSelector make(const edm::ParameterSet &cfg) {
        return TrackingParticleSelector(cfg.getParameter<double>("ptMin"),
                                        cfg.getParameter<double>("ptMax"),
                                        cfg.getParameter<double>("minRapidity"),
                                        cfg.getParameter<double>("maxRapidity"),
                                        cfg.getParameter<double>("tip"),
                                        cfg.getParameter<double>("lip"),
                                        cfg.getParameter<int>("minHit"),
                                        cfg.getParameter<bool>("signalOnly"),
                                        cfg.getParameter<bool>("intimeOnly"),
                                        cfg.getParameter<bool>("chargedOnly"),
                                        cfg.getParameter<bool>("stableOnly"),
                                        cfg.getParameter<std::vector<int>>("pdgId"),
                                        cfg.getParameter<bool>("invertRapidityCut"),
                                        cfg.getParameter<double>("minPhi"),
                                        cfg.getParameter<double>("maxPhi"));
      }
    };

  }  // namespace modules
}  // namespace reco

#endif
