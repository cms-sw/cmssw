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
  
  // Compute SimHit distance from strip
  float simHitDistFromStrip(const RPCDetId* rpcId,
  			   RPCId firstStrip,
  			   const PSimHit& hit);
  
  
};



#endif




