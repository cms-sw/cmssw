// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

  /**
   * This is a very simple test analyzer to test the update of a track with
   * a vertex constraint with the Kalman filter.
   */

class KVFTrackUpdate : public edm::EDAnalyzer {
public:
  explicit KVFTrackUpdate(const edm::ParameterSet&);
  ~KVFTrackUpdate() override;
  
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  void beginJob() override;
  void endJob() override;

private:

  edm::EDGetTokenT<reco::TrackCollection>	 token_tracks; 
  edm::EDGetTokenT<reco::BeamSpot> 	 token_beamSpot; 

};
