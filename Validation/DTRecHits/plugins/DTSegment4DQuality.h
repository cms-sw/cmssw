#ifndef Validation_DTSegment4D_H
#define Validation_DTSegment4D_H

/** \class DTSegment4DQuality
 *  Basic analyzer class which accesses 4D DTSegments
 *  and plots resolution comparing reconstructed and simulated quantities
 *
 *  Only true 4D segments are considered.  
 *  Station 4 segments are not looked at.
 *  FIXME: Add flag to consider also
 *  segments with only phi view? Possible bias?
 * 
 *  Residual/pull plots are filled for the reco segment with alpha closest
 *  to the simulated muon direction (defined from muon simhits in the chamber).
 *  
 *  Efficiencies are defined as reconstructed 4D segments with alpha, beta, x, y,
 *  within 5 sigma relative to the sim muon, with sigmas specified in the config.
 *  Note that loss of even only one of the two views is considered as inefficiency!
 *
 *  \author S. Bolognesi and G. Cerminara - INFN Torino
 */

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "Histograms.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

#include <vector>
#include <map>
#include <string>

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}

class TFile;

class DTSegment4DQuality : public edm::EDAnalyzer {
public:
  /// Constructor
  DTSegment4DQuality(const edm::ParameterSet& pset);

  /// Destructor
  ~DTSegment4DQuality() override;

  // Operations

  /// Perform the real analysis
  void analyze(const edm::Event & event, const edm::EventSetup& eventSetup) override;

  void beginRun(const edm::Run& iRun, const edm::EventSetup &setup) override;
  
  // Write the histos to file
  void endJob() override;
  void endLuminosityBlock(edm::LuminosityBlock const& lumiSeg,
			  edm::EventSetup const& c) override;

protected:

private: 

  // The file which will store the histos
  //TFile *theFile;
  // Switch for debug output
  bool debug;
  // Root file name
  std::string rootFileName;
  //Labels to read from event
  edm::InputTag simHitLabel;
  edm::InputTag segment4DLabel;
  edm::EDGetTokenT<edm::PSimHitContainer> simHitToken_;
  edm::EDGetTokenT<DTRecSegment4DCollection> segment4DToken_;
  //Sigma resolution on position
  double sigmaResX;
  double sigmaResY;
  //Sigma resolution on angle
  double sigmaResAlpha;
  double sigmaResBeta;

  HRes4DHit *h4DHit;
  HRes4DHit *h4DHit_W0;
  HRes4DHit *h4DHit_W1;
  HRes4DHit *h4DHit_W2;
  HRes4DHit *h4DHitWS[3][4];

  HEff4DHit *hEff_All;
  HEff4DHit *hEff_W0;
  HEff4DHit *hEff_W1;
  HEff4DHit *hEff_W2;
  HEff4DHit *hEffWS[3][4];

  DQMStore* dbe_;
  bool doall;
  bool local;
};

#endif
