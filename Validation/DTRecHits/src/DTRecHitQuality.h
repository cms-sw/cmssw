#ifndef Validation_DTRecHits_H
#define Validation_DTRecHits_H

/** \class DTRecHitQuality
 *  Basic analyzer class which accesses 1D DTRecHits
 *  and plot resolution comparing reconstructed and simulated quantities
 *
 *  $Date: 2006/03/22 16:15:36 $
 *  $Revision: 1.3 $
 *  \author G. Cerminara - INFN Torino
 */

#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"


#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "DataFormats/DTRecHit/interface/DTRecHitCollection.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment2DCollection.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"


#include <vector>
#include <map>
#include <string>

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}

class PSimHit;
class TFile;
class DTLayer;
class DTWireId;
class DTGeometry;

class DTRecHitQuality : public edm::EDAnalyzer {
public:
  /// Constructor
  DTRecHitQuality(const edm::ParameterSet& pset);

  /// Destructor
  virtual ~DTRecHitQuality();

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
  std::string simHitLabel;
  std::string recHitLabel;
  std::string segment2DLabel;
  std::string segment4DLabel;

  // Switches for analysis at various steps
  bool doStep1;
  bool doStep2;
  bool doStep3;

  // Map simhits per wireId
  std::map<DTWireId,  std::vector<PSimHit> > 
  mapSimHitsPerWire(const edm::PSimHitContainer* simhits);

  // Return a map between DTRecHit1DPair and wireId
  std::map<DTWireId, std::vector<DTRecHit1DPair> >
  map1DRecHitsPerWire(const DTRecHitCollection* dt1DRecHitPairs);

  // Return a map between DTRecHit1D and wireId
  std::map<DTWireId, std::vector<DTRecHit1D> >
  map1DRecHitsPerWire(const DTRecSegment2DCollection* segment2Ds);

  // Return a map between DTRecHit1D and wireId
  std::map<DTWireId, std::vector<DTRecHit1D> >
  map1DRecHitsPerWire(const DTRecSegment4DCollection* segment4Ds);

  // Find the mu simhit among a collection of simhits
  const PSimHit* findMuSimHit(const std::vector<PSimHit>& hits);

  // Compute SimHit distance from wire (cm)
  float simHitDistFromWire(const DTLayer* layer,
			   DTWireId wireId,
			   const PSimHit& hit);

  // Find the RecHit closest to the muon SimHit
//   const DTRecHit1DPair* 
//   findBestRecHit(const DTLayer* layer,
// 		 DTWireId wireId,
// 		 const std::vector<DTRecHit1DPair>& recHits,
// 		 const float simHitDist);
  template  <typename type>
  const type* 
  findBestRecHit(const DTLayer* layer,
				  DTWireId wireId,
				  const std::vector<type>& recHits,
				  const float simHitDist);




  // Compute the distance from wire (cm) of a hits in a DTRecHit1DPair
  float recHitDistFromWire(const DTRecHit1DPair& hitPair, const DTLayer* layer);
  // Compute the distance from wire (cm) of a hits in a DTRecHit1D
  float recHitDistFromWire(const DTRecHit1D& recHit, const DTLayer* layer);

  // Return the error on the measured (cm) coordinate
  float recHitPositionError(const DTRecHit1DPair& recHit);
  float recHitPositionError(const DTRecHit1D& recHit);


  // Does the real job
  template  <typename type>
  void compute(const DTGeometry *dtGeom,
	       std::map<DTWireId, std::vector<PSimHit> > simHitsPerWire,
	       std::map<DTWireId, std::vector<type> > recHitsPerWire,
	       int step);

};
#endif




