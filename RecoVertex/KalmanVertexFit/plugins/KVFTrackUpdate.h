// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"

/**
   * This is a very simple test analyzer to test the update of a track with
   * a vertex constraint with the Kalman filter.
   */

class KVFTrackUpdate : public edm::one::EDAnalyzer<> {
public:
  explicit KVFTrackUpdate(const edm::ParameterSet&);
  ~KVFTrackUpdate() override;

  void analyze(const edm::Event&, const edm::EventSetup&) override;

  void beginJob() override;
  void endJob() override;

private:
  const edm::ESGetToken<TransientTrackBuilder, TransientTrackRecord> estoken_TTB;
  edm::EDGetTokenT<reco::TrackCollection> token_tracks;
  edm::EDGetTokenT<reco::BeamSpot> token_beamSpot;
};
