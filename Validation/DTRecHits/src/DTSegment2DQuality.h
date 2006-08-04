#ifndef Validation_DTSegment2D_H
#define Validation_DTSegment2D_H

/** \class DTSegment2DQuality
 *  Basic analyzer class which accesses 2D DTSegments
 *  and plot resolution comparing reconstructed and simulated quantities
 *
 *  $Date: July 2006$
 *  $Revision: 1.0$
 *  \author S. Bolognesi and G. Cerminara - INFN Torino
 */

#include "FWCore/Framework/interface/EDAnalyzer.h"

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
  TFile *theFile;
  // Switch for debug output
  bool debug;
  // Root file name
  std::string rootFileName;
  //Labels to read from event
  std::string simHitLabel;
  std::string segment2DLabel;
  //Sigma resolution on position
  double sigmaResPos;
  //Sigma resolution on angle
  double sigmaResAngle;


};
#endif
