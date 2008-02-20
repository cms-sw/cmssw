// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

  /**
   * This is a very simple test analyzer to test the update of a track with
   * a vertex constraint with the Kalman filter.
   */

class KVFTrackUpdate : public edm::EDAnalyzer {
public:
  explicit KVFTrackUpdate(const edm::ParameterSet&);
  ~KVFTrackUpdate();
  
  virtual void analyze(const edm::Event&, const edm::EventSetup&);

  virtual void beginJob(edm::EventSetup const&);
  virtual void endJob();

private:

  edm::InputTag trackLabel_, beamSpotLabel;

};
