#ifndef Validation_DTRecHits_H
#define Validation_DTRecHits_H

/** \class DTRecHitQuality
 *  Basic analyzer class which accesses 1D DTRecHits
 *  and plot resolution comparing reconstructed and simulated quantities
 *
 *  $Date: 2010/09/17 07:48:11 $
 *  $Revision: 1.10 $
 *  \author G. Cerminara - INFN Torino
 */

#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "DataFormats/DTRecHit/interface/DTRecHitCollection.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment2DCollection.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"
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
  edm::InputTag simHitLabel;
  edm::InputTag recHitLabel;
  edm::InputTag segment2DLabel;
  edm::InputTag segment4DLabel;

  // Switches for analysis at various steps
  bool doStep1;
  bool doStep2;
  bool doStep3;
  bool local;
  // Return a map between DTRecHit1DPair and wireId
  std::map<DTWireId, std::vector<DTRecHit1DPair> >
    map1DRecHitsPerWire(const DTRecHitCollection* dt1DRecHitPairs);

  // Return a map between DTRecHit1D and wireId
  std::map<DTWireId, std::vector<DTRecHit1D> >
    map1DRecHitsPerWire(const DTRecSegment2DCollection* segment2Ds);

  // Return a map between DTRecHit1D and wireId
  std::map<DTWireId, std::vector<DTRecHit1D> >
    map1DRecHitsPerWire(const DTRecSegment4DCollection* segment4Ds);

  // Compute SimHit distance from wire (cm)
  float simHitDistFromWire(const DTLayer* layer,
                           DTWireId wireId,
                           const PSimHit& hit);

  // Compute SimHit impact angle (in direction perp to wire)
  float simHitImpactAngle(const DTLayer* layer,
			   DTWireId wireId,
			   const PSimHit& hit);

  // Compute SimHit distance from FrontEnd
  float simHitDistFromFE(const DTLayer* layer,
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
  //HRes1DHit * hRes_S1RPhi;
  HRes1DHit *hRes_S1RPhi;  // RecHits, 1. step, RPh
  HRes1DHit *hRes_S2RPhi;     // RecHits, 2. step, RPhi
  HRes1DHit *hRes_S3RPhi;     // RecHits, 3. step, RPhi

  HRes1DHit *hRes_S1RZ;         // RecHits, 1. step, RZ
  HRes1DHit *hRes_S2RZ;	    // RecHits, 2. step, RZ
  HRes1DHit *hRes_S3RZ;	    // RecHits, 3. step, RZ

  HRes1DHit *hRes_S1RZ_W0;   // RecHits, 1. step, RZ, wheel 0
  HRes1DHit *hRes_S2RZ_W0;   // RecHits, 2. step, RZ, wheel 0
  HRes1DHit *hRes_S3RZ_W0;   // RecHits, 3. step, RZ, wheel 0

  HRes1DHit *hRes_S1RZ_W1;   // RecHits, 1. step, RZ, wheel +-1
  HRes1DHit *hRes_S2RZ_W1;   // RecHits, 2. step, RZ, wheel +-1
  HRes1DHit *hRes_S3RZ_W1;   // RecHits, 3. step, RZ, wheel +-1

  HRes1DHit *hRes_S1RZ_W2;   // RecHits, 1. step, RZ, wheel +-2
  HRes1DHit *hRes_S2RZ_W2;   // RecHits, 2. step, RZ, wheel +-2
  HRes1DHit *hRes_S3RZ_W2;   // RecHits, 3. step, RZ, wheel +-2

  HRes1DHit *hRes_S1RPhi_W0;   // RecHits, 1. step, RPhi, wheel 0
  HRes1DHit *hRes_S2RPhi_W0;   // RecHits, 2. step, RPhi, wheel 0
  HRes1DHit *hRes_S3RPhi_W0;   // RecHits, 3. step, RPhi, wheel 0

  HRes1DHit *hRes_S1RPhi_W1;   // RecHits, 1. step, RPhi, wheel +-1
  HRes1DHit *hRes_S2RPhi_W1;   // RecHits, 2. step, RPhi, wheel +-1
  HRes1DHit *hRes_S3RPhi_W1;   // RecHits, 3. step, RPhi, wheel +-1

  HRes1DHit *hRes_S1RPhi_W2;   // RecHits, 1. step, RPhi, wheel +-2
  HRes1DHit *hRes_S2RPhi_W2;   // RecHits, 2. step, RPhi, wheel +-2
  HRes1DHit *hRes_S3RPhi_W2;   // RecHits, 3. step, RPhi, wheel +-2

  HEff1DHit *hEff_S1RPhi;     // RecHits, 1. step, RPhi
  HEff1DHit *hEff_S2RPhi;     // RecHits, 2. step, RPhi
  HEff1DHit *hEff_S3RPhi;     // RecHits, 3. step, RPhi

  HEff1DHit *hEff_S1RZ;         // RecHits, 1. step, RZ
  HEff1DHit *hEff_S2RZ;	    // RecHits, 2. step, RZ
  HEff1DHit *hEff_S3RZ;	    // RecHits, 3. step, RZ

  HEff1DHit *hEff_S1RZ_W0;   // RecHits, 1. step, RZ, wheel 0
  HEff1DHit *hEff_S2RZ_W0;   // RecHits, 2. step, RZ, wheel 0
  HEff1DHit *hEff_S3RZ_W0;   // RecHits, 3. step, RZ, wheel 0

  HEff1DHit *hEff_S1RZ_W1;   // RecHits, 1. step, RZ, wheel +-1
  HEff1DHit *hEff_S2RZ_W1;   // RecHits, 2. step, RZ, wheel +-1
  HEff1DHit *hEff_S3RZ_W1;   // RecHits, 3. step, RZ, wheel +-1

  HEff1DHit *hEff_S1RZ_W2;   // RecHits, 1. step, RZ, wheel +-2
  HEff1DHit *hEff_S2RZ_W2;   // RecHits, 2. step, RZ, wheel +-2
  HEff1DHit *hEff_S3RZ_W2;   // RecHits, 3. step, RZ, wheel +-2
  DQMStore* dbe_;
  bool doall;
};
#endif




