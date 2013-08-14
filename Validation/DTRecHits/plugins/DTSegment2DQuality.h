#ifndef Validation_DTSegment2D_H
#define Validation_DTSegment2D_H

/** \class DTSegment2DQuality
 *  Basic analyzer class which accesses 2D DTSegments
 *  and plot resolution comparing reconstructed and simulated quantities
 *
 *  $Date: 2010/09/13 09:49:18 $
 *  $Revision: 1.4 $
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

class DTSegment2DQuality : public edm::EDAnalyzer {
public:
  /// Constructor
  DTSegment2DQuality(const edm::ParameterSet& pset);

  /// Destructor
  virtual ~DTSegment2DQuality();

  // Operations

  /// Perform the real analysis
  void analyze(const edm::Event & event, const edm::EventSetup& eventSetup);
  // Write the histos to file
  void endJob();

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
  edm::InputTag segment2DLabel;
  //Sigma resolution on position
  double sigmaResPos;
  //Sigma resolution on angle
  double sigmaResAngle;

  HRes2DHit *h2DHitRPhi;
  HRes2DHit *h2DHitRZ;
  HRes2DHit *h2DHitRZ_W0;
  HRes2DHit *h2DHitRZ_W1;
  HRes2DHit *h2DHitRZ_W2;

  HEff2DHit *h2DHitEff_RPhi;
  HEff2DHit *h2DHitEff_RZ;
  HEff2DHit *h2DHitEff_RZ_W0;
  HEff2DHit *h2DHitEff_RZ_W1;
  HEff2DHit *h2DHitEff_RZ_W2;
  DQMStore* dbe_;
};
#endif
