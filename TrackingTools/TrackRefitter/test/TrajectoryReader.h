#ifndef TrackingTools_TrackRefitter_TrajectoryReader_H
#define TrackingTools_TrackRefitter_TrajectoryReader_H

/** \class TrajectoryReader
 *  No description available.
 *
 *  $Date: $
 *  $Revision: $
 *  \author R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 */
// Base Class Headers
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/ParameterSet/interface/InputTag.h"

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}

class TFile;
class TH1F;
class TH2F;
class Trajectory;

#include <vector>

class TrajectoryReader: public edm::EDAnalyzer {

 public:
  typedef std::vector<Trajectory> Trajectories;

 public:
  /// Constructor
  TrajectoryReader(const edm::ParameterSet& pset);

  /// Destructor
  virtual ~TrajectoryReader();

  void analyze(const edm::Event & event, const edm::EventSetup& eventSetup);

  // Operations
  void beginJob(const edm::EventSetup&);
  void endJob();
  
protected:

private:

  edm::InputTag theInputLabel;
  TFile *theFile;
  std::string theRootFileName;
  
  TH1F *hDPtIn;
  TH1F *hDPtOut;
};
#endif
