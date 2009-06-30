#ifndef Validation_DTSegment2DSLPhi_H
#define Validation_DTSegment2DSLPhi_H

/** \class DTSegment2DSLPhiQuality
 *  Basic analyzer class which accesses 2D DTSegments reconstructed with both SL Phi
 *  and plot resolution comparing reconstructed and simulated quantities
 *
 *  $Date: 2007/06/08 15:50:31 $
 *  $Revision: 1.2 $
 *  \author S. Bolognesi and G. Cerminara - INFN Torino
 */

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "Histograms.h"

#include <vector>
#include <map>
#include <string>

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
  // Write the histos to file
  void endJob();

protected:

private: 

  // The file which will store the histos
  TFile *theFile;
  // Switch for debug output
  bool debug;
  // Root file name
  std::string rootFileName;
  //Labels to read from event
  std::string simHitLabel;
  std::string segment4DLabel;
  //Sigma resolution on position
  double sigmaResPos;
  //Sigma resolution on angle
  double sigmaResAngle;

  HRes2DHit *h2DHitSuperPhi;
  HEff2DHit *h2DHitEff_SuperPhi;
};
#endif
