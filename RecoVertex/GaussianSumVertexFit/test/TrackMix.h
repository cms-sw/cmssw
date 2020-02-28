/**\class TrackMix TrackMix.cc RecoVertex/TrackMix/src/TrackMix.cc

 Description: Simple test to see that reco::Vertex can store tracks from different
 Collections and of different types in the same vertex.
*/

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include <TFile.h>

/**
   * This is a very simple test analyzer mean to test the KalmanVertexFitter
   */

class TrackMix : public edm::EDAnalyzer {
public:
  explicit TrackMix(const edm::ParameterSet&);
  ~TrackMix();

  virtual void analyze(const edm::Event&, const edm::EventSetup&);

  virtual void beginJob();
  virtual void endJob();

private:
  edm::ParameterSet theConfig;
  edm::EDGetTokenT<edm::View<reco::Track> > token_gsf, token_ckf;
};
