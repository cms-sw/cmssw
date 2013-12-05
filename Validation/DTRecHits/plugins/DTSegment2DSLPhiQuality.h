#ifndef Validation_DTSegment2DSLPhi_H
#define Validation_DTSegment2DSLPhi_H

/** \class DTSegment2DSLPhiQuality
 *  Basic analyzer class which accesses 2D DTSegments reconstructed with both SL Phi
 *  and plot resolution comparing reconstructed and simulated quantities
 *
 *  \author S. Bolognesi and G. Cerminara - INFN Torino
 */

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "Histograms.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "TStyle.h"
#include <vector>
#include <map>
#include <string>
#include "FWCore/Utilities/interface/InputTag.h"

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}

class TFile;

class DTSegment2DSLPhiQuality : public edm::EDAnalyzer {
public:
  /// Constructor
  DTSegment2DSLPhiQuality(const edm::ParameterSet& pset);

  /// Destructor
  virtual ~DTSegment2DSLPhiQuality();

  // Operations

  /// Perform the real analysis
  void analyze(const edm::Event & event, const edm::EventSetup& eventSetup);

  virtual void beginRun(const edm::Run& iRun, const edm::EventSetup &setup);

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
  double sigmaResPos;
  //Sigma resolution on angle
  double sigmaResAngle;

  HRes2DHit *h2DHitSuperPhi;
  HEff2DHit *h2DHitEff_SuperPhi;
  DQMStore* dbe_;
  bool doall;
  bool local;
  //  TStyle * mystyle;
};
#endif
