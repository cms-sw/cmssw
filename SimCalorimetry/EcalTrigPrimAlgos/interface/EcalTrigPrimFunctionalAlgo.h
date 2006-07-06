#ifndef EcalTrigPrimFunctionalAlgo_h
#define EcalTrigPrimFunctionalAlgo_h

#include <vector>
#include <map>
#include <utility>

class TTree;

class EBDataFrame;
class EEDataFrame;
class EcalTrigTowerDetId;
class ETPCoherenceTest;
class EcalTriggerPrimitiveSample;
class EcalBarrelTopology;
 
#include "Geometry/CaloTopology/interface/EcalTrigTowerConstituentsMap.h"
#include "SimCalorimetry/EcalTrigPrimAlgos/interface/EcalBarrelFenixStrip.h"
#include "SimCalorimetry/EcalTrigPrimAlgos/interface/EcalBarrelFenixTcp.h"
//#include "SimCalorimetry/EcalTrigPrimAlgos/interface/EcalEndcapFenixTcp.h"
#include "DataFormats/EcalDigi/interface/EBDataFrame.h"
#include "DataFormats/EcalDigi/interface/EEDataFrame.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

using namespace tpg; //FIXME

/** Main Algo for Ecal trigger primitives. */
class EcalTrigPrimFunctionalAlgo
{  
public:
  
  //  typedef PRecDet<EcalTrigPrim> precdet;

  explicit EcalTrigPrimFunctionalAlgo(const edm::EventSetup & setup,int binofmax, int nrsamples);
  EcalTrigPrimFunctionalAlgo(const edm::EventSetup & setup, TTree *tree, int binofmax, int nrsamples);
  virtual ~EcalTrigPrimFunctionalAlgo();

  /** this actually calculates the trigger primitives (from Digis) */

  void run(const EBDigiCollection* ebdcol, const EEDigiCollection* eedcol, EcalTrigPrimDigiCollection & result, int fgvbMinEn);

private:

   void init(const edm::EventSetup & setup);

 //  vector<string> theBaseNames;


  float threshold;
  
  void fillBarrel(const EcalTrigTowerDetId & coarser, const EBDataFrame & frame);

  void fillEndcap(const EcalTrigTowerDetId & coarser, const EEDataFrame & frame);

  // create stripnr
  int createStripNr(const EBDetId& cryst);

  int calculateTTF(const int en);

 //--------------------


  typedef std::map<EcalTrigTowerDetId,std::vector<std::vector<EBDataFrame> >,std::less<EcalTrigTowerDetId> > SUMVB;
  //  typedef std::map<EcalTrigTowerDetId,std::vector<std::vector<EEDataFrame> >,std::less<EcalTrigTowerDetId> > SUMVE;

  // typedef std::map<CellID,CaloTimeSample,less<CellID> > SUM;
  // temporary, waiting for pseudostrip geometry
  // SUMVE sumEndcap_;
  // this contains for each trigger tower, first the summed energies, then all EEDataFrames beloonging to this tower
   typedef std::map<EcalTrigTowerDetId,std::vector<int> > SUMVE;
   SUMVE sumEndcap_;
   typedef std::map<EcalTrigTowerDetId,std::vector<EEDataFrame> > MAPE;
   MAPE mapEndcap_;

  /** map of (coarse granularity) cell to the CaloTimeSample objects
      associated to this cell for the EcalBarrel. */
  SUMVB sumBarrel_; 


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

  //  EcalEndcapFenixTcp eetcp_;

   edm::ESHandle<EcalTrigTowerConstituentsMap> eTTmap_;

  // for debugging
  ETPCoherenceTest *cTest_;

  //for validation
  bool valid_;
  TTree * valTree_;

  int binOfMaximum_;
  int nrSamplesToWrite_;
};

#endif
