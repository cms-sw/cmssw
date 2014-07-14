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
			     double tip,double lip,int minHit, bool signalOnly, bool chargedOnly, bool stableOnly,
			     const std::vector<int>& pdgId = std::vector<int>()) :
    ptMin_( ptMin ), minRapidity_( minRapidity ), maxRapidity_( maxRapidity ),
    tip_( tip ), lip_( lip ), minHit_( minHit ), signalOnly_(signalOnly), chargedOnly_(chargedOnly), stableOnly_(stableOnly), pdgId_( pdgId ) { }

  /// Operator() performs the selection: e.g. if (tPSelector(tp)) {...}
  bool operator()( const TrackingParticle & tp ) const {
    if (chargedOnly_ && tp.threeCharge()==0) return false;//select only if charge!=0
    bool testId = false;
    unsigned int idSize = pdgId_.size();
    if (idSize==0) testId = true;
    else for (unsigned int it=0;it!=idSize;++it){
      if (tp.pdgId()==pdgId_[it]) testId = true;
    }
    bool signal = true;
    if (signalOnly_) signal = (tp.eventId().bunchCrossing()== 0 && tp.eventId().event() == 0); // signal only means no PU particles
    // select only stable particles
    bool stable = true;
    if (stableOnly_) {
      if (!signal) {
	stable = false; // we are not interested into PU particles among the stable ones
      } else {
	for( TrackingParticle::genp_iterator j = tp.genParticle_begin(); j != tp.genParticle_end(); ++ j ) {
	  if (j->get()==0 || j->get()->status() != 1) {
	    stable = false; break;
	  }
	}
       // test for remaining unstabled due to lack of genparticle pointer
       if( stable  && tp.status() == -99 &&
          (std::abs(tp.pdgId()) != 11 && std::abs(tp.pdgId()) != 13 && std::abs(tp.pdgId()) != 211 &&
           std::abs(tp.pdgId()) != 321 && std::abs(tp.pdgId()) != 2212 && std::abs(tp.pdgId()) != 3112 &&
           std::abs(tp.pdgId()) != 3222 && std::abs(tp.pdgId()) != 3312 && std::abs(tp.pdgId()) != 3334)) stable = 0;
      }
    }

    auto etaOk = [&](const TrackingParticle::Vector& p)->bool{ float eta= etaFromXYZ(p.x(),p.y(),p.z()); return (eta>= minRapidity_) & (eta<=maxRapidity_);};
    return (
            (testId & signal & stable) &&
 	    tp.numberOfTrackerLayers() >= minHit_ &&
            std::abs(tp.vertex().z()) <= lip_ &&
	    tp.momentum().perp2() >= ptMin_*ptMin_ &&
	    tp.vertex().perp2() <= tip_*tip_ &&
            etaOk(tp.momentum())
	    );
  }

private:
  double ptMin_;
  float minRapidity_;
  float maxRapidity_;
  double tip_;
  double lip_;
  int    minHit_;
  bool signalOnly_;
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
	  cfg.getParameter<bool>( "chargedOnly" ),
	  cfg.getParameter<bool>( "stableOnly" ),
	cfg.getParameter<std::vector<int> >( "pdgId" ));
      }
    };

  }
}

#endif
