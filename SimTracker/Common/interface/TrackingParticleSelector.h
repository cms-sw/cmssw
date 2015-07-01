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
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/Math/interface/PtEtaPhiMass.h"

class TrackingParticleSelector {

public:
  TrackingParticleSelector(){}
  TrackingParticleSelector ( double ptMin,double minRapidity,double maxRapidity,
			     double tip,double lip,int minHit, bool signalOnly, bool intimeOnly, bool chargedOnly, bool stableOnly,
			     const std::vector<int>& pdgId = std::vector<int>()) :
    ptMin2_( ptMin*ptMin ), minRapidity_( minRapidity ), maxRapidity_( maxRapidity ),
    tip2_( tip*tip ), lip_( lip ), minHit_( minHit ), signalOnly_(signalOnly), intimeOnly_(intimeOnly), chargedOnly_(chargedOnly), stableOnly_(stableOnly), pdgId_( pdgId ) { }

  /// Operator() performs the selection: e.g. if (tPSelector(tp)) {...}
  bool operator()( const TrackingParticle & tp ) const {
    // signal only means no PU particles
    if (signalOnly_ && !(tp.eventId().bunchCrossing()== 0 && tp.eventId().event() == 0)) return false;
    // intime only means no OOT PU particles
    if (intimeOnly_ && !(tp.eventId().bunchCrossing()==0)) return false;

    auto pdgid = tp.pdgId();
    if(!pdgId_.empty()) {
      bool testId = false;
      for(auto id: pdgId_) {
        if(id == pdgid) { testId = true; break;}
      }
      if(!testId) return false;
    }

    if (chargedOnly_ && tp.charge()==0) return false;//select only if charge!=0

    // select only stable particles
    if (stableOnly_) {
      for( TrackingParticle::genp_iterator j = tp.genParticle_begin(); j != tp.genParticle_end(); ++ j ) {
        if (j->get()==0 || j->get()->status() != 1) {
          return false;
        }
      }
      // test for remaining unstabled due to lack of genparticle pointer
      if( tp.status() == -99 &&
          (std::abs(pdgid) != 11 && std::abs(pdgid) != 13 && std::abs(pdgid) != 211 &&
           std::abs(pdgid) != 321 && std::abs(pdgid) != 2212 && std::abs(pdgid) != 3112 &&
           std::abs(pdgid) != 3222 && std::abs(pdgid) != 3312 && std::abs(pdgid) != 3334))
        return false;
    }

    auto etaOk = [&](const TrackingParticle& p)->bool{ float eta= etaFromXYZ(p.px(),p.py(),p.pz()); return (eta>= minRapidity_) & (eta<=maxRapidity_);};
    return (
 	    tp.numberOfTrackerLayers() >= minHit_ &&
	    tp.p4().perp2() >= ptMin2_ &&
            etaOk(tp) &&
            std::abs(tp.vertex().z()) <= lip_ &&   // vertex last to avoid to load it if not striclty necessary...
	    tp.vertex().perp2() <= tip2_
	    );
  }

private:
  double ptMin2_;
  float minRapidity_;
  float maxRapidity_;
  double tip2_;
  double lip_;
  int    minHit_;
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

    template<>
    struct ParameterAdapter<TrackingParticleSelector> {
      static TrackingParticleSelector make( const edm::ParameterSet & cfg, edm::ConsumesCollector & iC ) {
	return TrackingParticleSelector(
 	  cfg.getParameter<double>( "ptMin" ),
	  cfg.getParameter<double>( "minRapidity" ),
	  cfg.getParameter<double>( "maxRapidity" ),
	  cfg.getParameter<double>( "tip" ),
	  cfg.getParameter<double>( "lip" ),
	  cfg.getParameter<int>( "minHit" ),
	  cfg.getParameter<bool>( "signalOnly" ),
          cfg.getParameter<bool>( "intimeOnly" ),
	  cfg.getParameter<bool>( "chargedOnly" ),
	  cfg.getParameter<bool>( "stableOnly" ),
	cfg.getParameter<std::vector<int> >( "pdgId" ));
      }
    };

  }
}

#endif
