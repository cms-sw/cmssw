
#ifndef Validation_RPCRecHits_H
#define Validation_RPCRecHits_H

/*
 * \class RPCRecHitQuality
 *  Basic analyzer class which accesses 1D RPCRecHits
 *  and plot resolution comparing reconstructed and simulated quantities
 *
 *  $Date: 2007/01/12 12:34:38 $
 *  $Revision: 1.2 $
 *  \author D. Pagano - Dip. Fis. Nucl. e Teo. & INFN Pavia
 */



#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/RPCRecHit/interface/RPCRecHitCollection.h"


//#include <vector>
//#include <map>
//#include <string>


namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}



class PSimHit;
class TFile;
class RPCRoll;
class RPCCluster;
class RPCGeometry;



class RPCRecHitQuality : public edm::EDAnalyzer {

public:

  // Constructor
  RPCRecHitQuality(const edm::ParameterSet&);

  // Destructor
  virtual ~RPCRecHitQuality();

  void arrange(MixCollection<PSimHit> & simHits,
               RPCDigiCollection & rpcDigis);

  // Perform the real analysis
  void analyze(const edm::Event & event, const edm::EventSetup& eventSetup);


   // Write the histos to file
  void endJob();

    
private: 
  

  // The file which will store the histos
  TFile *theFile;

 
  // Root file name
  std::string rootFileName;
  std::string simHitLabel;
  std::string recHitLabel;
  std::string digiLabel;

  
  /*
  // Return a map between RPCRecHitPair and rpcId 
  std::map<RPCDetId, std::vector<RPCRecHit> >
  map1DRecHitsPerStrip(const RPCRecHitCollection* RPCRecHitPairs);

  
  // Compute SimHit distance from strip
  float simHitDistFromStrip(const RPCRoll& roll,
  			    const RPCCluster& cl,
  			    const PSimHit& hit);
  

  // Find the RecHit closest to the muon SimHit
  template  <typename type>
  const type* 
  findBestRecHit(const RPCRoll& roll,
  		 const RPCCluster& cl,
		 const std::vector<type>& recHits,
		 const float simHitDist);




  // Compute the distance from strip (cm) of a hits
  float recHitDistFromStrip(const RPCRecHit& hitPair, 
                            const RPCCluster& cl);
  

  // Return the error on the measured (cm) coordinate
  float recHitPositionError(const RPCRecHit& recHit);
  


  // Does the real job
  template  <typename type>
  void compute(const RPCGeometry *rpcGeom,
	       std::map<RPCDetId, std::vector<PSimHit> > simHitsPerStrip,
	       std::map<RPCDetId, std::vector<type> > recHitsPerStrip,
	       int step);


  
  */
  
};

  

#endif




