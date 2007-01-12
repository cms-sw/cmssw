#ifndef Validation_RPCRecHits_H
#define Validation_RPCRecHits_H

/*
 * \class RPCRecHitQuality
 *  Basic analyzer class which accesses 1D RPCRecHits
 *  and plot resolution comparing reconstructed and simulated quantities
 *
 *  $Date: 2006/12/19 18:16:22 $
 *  $Revision: 1.2 $
 *  \author D. Pagano - Dip. Fis. Nucl. e Teo. & INFN Pavia
 */



#include "FWCore/Framework/interface/EDAnalyzer.h"



#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"



#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/RPCRecHit/interface/RPCRecHitCollection.h"


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
class RPCRoll;
class RPCDetId;
class RPCGeometry;


class RPCRecHitQuality : public edm::EDAnalyzer {


  

public:
  /// Constructor
  RPCRecHitQuality(const edm::ParameterSet& pset);

  /// Destructor
  virtual ~RPCRecHitQuality();

  // Operations

  // Perform the real analysis
  void analyze(const edm::Event & event, const edm::EventSetup& eventSetup);

  // Write the histos to file
  void endJob();

  

protected:



private: 


  // The file which will store the histos
  TFile *theFile;
  
 
  // Root file name
  std::string rootFileName;
  std::string simHitLabel;
  std::string recHitLabel;
  


  // Return a map between RPCRecHitPair and rpcId 
 
  std::map<RPCDetId, std::vector<RPCRecHit> >
  map1DRecHitsPerRpcId(const RPCRecHitCollection* RPCRecHitPairs);
  
  // Compute SimHit distance from wire (cm)
  //float simHitDistFromWire(const DTLayer* layer,
  //			   DTWireId wireId,
  //			   const PSimHit& hit);
  
  // Find the RecHit closest to the muon SimHit
  //   const DTRecHit1DPair* 
  //   findBestRecHit(const DTLayer* layer,
  // 		 DTWireId wireId,
  // 		 const std::vector<DTRecHit1DPair>& recHits,
  // 		 const float simHitDist);

  //template  <typename type>
  //const type* 
  

  // Return the error on the measured (cm) coordinate
  //float recHitPositionError(const RPCRecHit1DPair& recHit);
  //float recHitPositionError(const RPCRecHit1D& recHit);

  // Does the real job
  //template  <typename type>
  //void compute(const RPCGeometry *dtGeom,
  //	       std::map<RPCStripId, std::vector<PSimHit> > simHitsPerStrip,
  //	       std::map<RPCStripId, std::vector<type> > recHitsPerStrip,
  //	       int step);

};



#endif




