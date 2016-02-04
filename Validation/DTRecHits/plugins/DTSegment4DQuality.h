#ifndef Validation_DTSegment4D_H
#define Validation_DTSegment4D_H

/** \class DTSegment4DQuality
 *  Basic analyzer class which accesses 4D DTSegments
 *  and plot resolution comparing reconstructed and simulated quantities
 *
 *  $Date: 2010/09/13 09:49:18 $
 *  $Revision: 1.7 $
 *  \author S. Bolognesi and G. Cerminara - INFN Torino
 */

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "Histograms.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/InputTag.h"

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
  virtual ~DTSegment4DQuality();

  // Operations

  /// Perform the real analysis
  void analyze(const edm::Event & event, const edm::EventSetup& eventSetup);
  // Write the histos to file
  void endJob();
  void endLuminosityBlock(edm::LuminosityBlock const& lumiSeg,
			  edm::EventSetup const& c);

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

  HEff4DHit *hEff_All;
  HEff4DHit *hEff_W0;
  HEff4DHit *hEff_W1;
  HEff4DHit *hEff_W2;
  DQMStore* dbe_;
  bool doall;
  bool local;
};

#endif
