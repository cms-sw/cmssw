#ifndef EcalTrigPrimFunctionalAlgo_h
#define EcalTrigPrimFunctionalAlgo_h

#include <vector>
#include <map>
#include <utility>

//class CellID;
//class CaloDataFrame;
class EBDataFrame;
class EEDataFrame;
class EcalTrigTowerDetId;
class ETPCoherenceTest;
class EcalTriggerPrimitiveSample;
class EcalBarrelTopology;
 
/* #include "CARF/SetUp/interface/SuId.h" */

/* #include "Utilities/UI/interface/Verbosity.h" */
/* #include "CARF/G3Event/interface/G3EventProxy.h"   */
/* #include "Utilities/Notification/interface/Observer.h" */
/* #include "CARF/G3PersistentReco/interface/PRecDet.h" */
#include "SimCalorimetry/EcalTrigPrimAlgos/interface/EcalBarrelFenixStrip.h"
#include "SimCalorimetry/EcalTrigPrimAlgos/interface/EcalBarrelFenixTcp.h"
#include "DataFormats/EcalDigi/interface/EBDataFrame.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "FWCore/Framework/interface/EventSetup.h"
using namespace tpg;

/** Main Algo for Ecal trigger primitives. */
class EcalTrigPrimFunctionalAlgo
{  
public:
  
  //  typedef PRecDet<EcalTrigPrim> precdet;

  explicit EcalTrigPrimFunctionalAlgo(const edm::EventSetup & setup);
  virtual ~EcalTrigPrimFunctionalAlgo();

  /** this actually calculates the trigger primitives (from Digis) */

  void run(const EBDigiCollection* ebdcol, EcalTrigPrimDigiCollection & result);

private:

    //  vector<string> theBaseNames;


  float threshold;
  
  void fillBarrel(const EcalTrigTowerDetId & coarser, const EBDataFrame & frame);


  // create stripnr
  int createStripNr(const EBDetId& cryst);

  //--------------------


 typedef std::map<EcalTrigTowerDetId,std::vector<std::vector<EBDataFrame> >,std::less<EcalTrigTowerDetId> > SUMV;



  /** map of (coarse granularity) cell to the CaloTimeSample objects
      associated to this cell for the EcalBarrel. */
  SUMV sumBarrel_; 


  /** number of 'strips' (crystals of same eta index) per trigger
      tower in ecal barrel */
  enum {ecal_barrel_strips_per_trigger_tower = 5};
  
  /** number of crystal per such 'strip' */
  enum {ecal_barrel_crystals_per_strip = 5};

  /** max number of crystals per pseudostrip in Endcap */
  enum {ecal_endcap_maxcrystals_per_strip = 5};

  /** max number of crystals per pseudostrip in Endcap */
  enum {ecal_endcap_maxstrips_per_tower = 5};

  //
   EcalBarrelFenixStrip * ebstrip_;

   EcalBarrelFenixTcp ebtcp_;

   EcalBarrelTopology* ebTopology_;

   // for debugging
   ETPCoherenceTest *cTest_;
};

#endif
