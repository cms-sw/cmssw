
#ifndef Validation_RPCRecHits_H
#define Validation_RPCRecHits_H

/*
 * \class RPCRecHitQuality
 *  Basic analyzer class which accesses 1D RPCRecHits
 *  and plot resolution comparing reconstructed and simulated quantities
 *
 *  $Date: 2007/01/30 13:59:53 $
 *  $Revision: 1.4 $
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

  
};

  

#endif




