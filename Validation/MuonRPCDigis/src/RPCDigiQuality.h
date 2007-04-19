#ifndef Validation_RPCDigis_H
#define Validation_RPCDigis_H

/*
 * \class RPCDigiQuality
 *  Basic analyzer class which accesses RPCDigis
 *  and plots of distance w.r.t. SimHits
 *
 *  $Date: 2007/04/16 11:31:01 $
 *  $Revision: 1.1 $
 *  \author M. Maggi - I.N.F.N. Bari
 */



#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"
#include "TH1F.h"
#include "TFolder.h"




namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}


//class PSimHit;
class TFile;
//class RPCRoll;

//class RPCGeometry;



class RPCDigiQuality : public edm::EDAnalyzer {

public:

  // Constructor
  RPCDigiQuality(const edm::ParameterSet&);

  // Destructor
  virtual ~RPCDigiQuality();

  
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
