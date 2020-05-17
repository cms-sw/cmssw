#ifndef TrackingTools_TrackRefitter_TrajectoryReader_H
#define TrackingTools_TrackRefitter_TrajectoryReader_H

/** \class TrajectoryReader
 *  No description available.
 *
 *  \author R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 */
// Base Class Headers
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}  // namespace edm

namespace reco {
  class Track;
}

class TFile;
class TH1F;
class TH2F;
class Trajectory;

#include <vector>

class TrajectoryReader : public edm::EDAnalyzer {
public:
  typedef std::vector<Trajectory> Trajectories;

public:
  /// Constructor
  TrajectoryReader(const edm::ParameterSet &pset);

  /// Destructor
  ~TrajectoryReader() override;

  void analyze(const edm::Event &event, const edm::EventSetup &eventSetup) override;

  // Operations
  void beginJob() override;
  void endJob() override;

protected:
  void printTrajectoryRecHits(const Trajectory &, const edm::ESHandle<GlobalTrackingGeometry> &) const;
  void printTrackRecHits(const reco::Track &, const edm::ESHandle<GlobalTrackingGeometry> &) const;

private:
  edm::InputTag theInputLabel;
  TFile *theFile;
  std::string theRootFileName;

  TH1F *hDPtIn;
  TH1F *hDPtOut;
  TH1F *hSuccess;
  TH1F *hNHitLost;
  TH1F *hFractionHitLost;
};
#endif
