#ifndef TrackingTools_TrackRefitter_TrackReader_H
#define TrackingTools_TrackRefitter_TrackReader_H

/** \class TrackReader
 *  No description available.
 *
 *  $Date: 2010/02/11 00:15:18 $
 *  $Revision: 1.3 $
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
}

namespace reco {class Track;}

class TFile;
class TH1F;
class TH2F;
class Trajectory;


#include <vector>

class TrackReader: public edm::EDAnalyzer {

 public:
  typedef std::vector<Trajectory> Trajectories;

 public:
  /// Constructor
  TrackReader(const edm::ParameterSet& pset);

  /// Destructor
  virtual ~TrackReader();

  void analyze(const edm::Event & event, const edm::EventSetup& eventSetup);

  // Operations
  void beginJob();
  void endJob();
  
protected:
  //  void printTrackRecHits(const reco::Track &, edm::ESHandle<GlobalTrackingGeometry>) const;
  
 private:
  edm::InputTag theInputLabel;

  std::string theTrackerRecHitBuilderName;
  std::string theMuonRecHitBuilderName;
};
#endif
