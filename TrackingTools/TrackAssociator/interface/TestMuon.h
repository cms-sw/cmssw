#ifndef TrackAssociator_TestMuon_h
#define TrackAssociator_TestMuon_h

#include "DataFormats/TrackReco/interface/TrackBase.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "TrackingTools/TrackAssociator/interface/TestMuonFwd.h"
#include "DataFormats/DetId/interface/DetId.h"

namespace reco {
   // class TestMuon : public TrackBase {
   class TestMuon : public Muon {
    public:
      /// default constructor
      TestMuon() { }
      
      /// constructor from fit parameters and error matrix
      // TestMuon( double chi2, double ndof,
      //		 const ParameterVector & par, double pt, const CovarianceMatrix & cov );
      
      /// Muon constructor
      TestMuon(  Charge, const LorentzVector &, const Point & = Point( 0, 0, 0 ) );
      
      // define containers
      struct MuonEnergy {
	 Double32_t had;   // energy deposited in HCAL
	 Double32_t em;    // energy deposited in ECAL
	 Double32_t ho;    // energy deposited in HO
      };
      
      // Sum-Et and Sum-Pt in cones of 0.1, 0.4, 0.7
      // CONE ENERGY (now)
      // radition, larger cone than 0.1
      struct MuonIsolation {
	 Double32_t hCalEt01;
	 Double32_t eCalEt01;
	 Double32_t hCalEt04;
	 Double32_t eCalEt04;
	 Double32_t hCalEt07;
	 Double32_t eCalEt07;
	 Double32_t trackSumPt01;
	 Double32_t trackSumPt04;
	 Double32_t trackSumPt07;
      };
      
      // matching information in local coordinates
      // correlation matrix ? Chi2? 
      // Surface position?
      // muon propagation
      struct MuonMatch {
	 Double32_t dX;      // X matching between track and segment
	 Double32_t dY;      // Y matching between track and segment
	 Double32_t dXErr;   // error in X matching
	 Double32_t dYErr;   // error in Y matching
	 Double32_t dXdZ;    // dX/dZ matching between track and segment
	 Double32_t dYdZ;    // dY/dZ matching between track and segment
	 Double32_t dXdZErr; // error in dX/dZ matching
	 Double32_t dYdZErr; // error in dY/dZ matching
	 DetId stationId;    // station ID
      };
      
      MuonEnergy calEnergy() const { return calEnergy_; }
      void setCalEnergy( const MuonEnergy& calEnergy ) { calEnergy_ = calEnergy; }
      
      MuonIsolation isolation() const { return isolation_;}
      void setIsolation( const MuonIsolation& isolation ) { isolation_ = isolation; }

      std::vector<MuonMatch> matches() const { return muMatches_;}
      void setMatches( const std::vector<MuonMatch>& matches ) { muMatches_ = matches; }
      
      int numberOfMatches() const { return muMatches_.size(); }
      double dX(int i) const { return muMatches_[i].dX; }
      double dY(int i) const { return muMatches_[i].dY; }
      double dXErr(int i) const { return muMatches_[i].dXErr; }
      double dYErr(int i) const { return muMatches_[i].dYErr; }
	   
      
      // const CaloTowerRefs& traversedTowers() {return traversedTowers_;}
      
    private:
      MuonEnergy calEnergy_;

      // vector of references to traversed towers. Could be useful for
      // correcting the missing transverse energy.  This assumes that
      // the CaloTowers will be kept in the AOD.  If not, we need something else.
      // CaloTowerRefs traversedTowers_;
      
      MuonIsolation isolation_;

      // Information on matching between tracks and segments
      std::vector<MuonMatch> muMatches_;

      // Vector of station IDs crossed by the track
      // (This is somewhat redundant with mu_Matches_ but allows
      // to see what segments were "missed".
      std::vector<DetId> crossedStationID_;

      // Still missing trigger information
   };
}

#endif
