
#ifndef Validation_RPCRecHits_H
#define Validation_RPCRecHits_H

/*
 * \class RPCRecHitQuality
 *  Basic analyzer class which accesses 1D RPCRecHits
 *  and plots residuals and pulls as function of some parameters
 *
 *  $Date: 2007/02/09 11:45:30 $
 *  $Revision: 1.6 $
 *  \author D. Pagano - Dip. Fis. Nucl. e Teo. & INFN Pavia
 */



#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/RPCRecHit/interface/RPCRecHitCollection.h"
#include "TH1F.h"
#include "TFolder.h"


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
  TH1F* ochamb[5][12][4][4];
  TH1F* test[5]; 
  TH1F* clchamb[5][12][4][4];
  TFolder* sec[5][12];
  TFolder* whe[5];
  TFolder* sta[5][12][4];	
  TFolder* lay[5][12][4][2];
};

  

#endif




