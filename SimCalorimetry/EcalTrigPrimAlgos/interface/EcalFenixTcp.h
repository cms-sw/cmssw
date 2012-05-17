#ifndef ECAL_FENIX_TCP_H
#define ECAL_FENIX_TCP_H

#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixEtTot.h>
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixBypassLin.h>
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixMaxof2.h>
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixTcpFormat.h>
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixFgvbEB.h>
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixTcpFgvbEE.h>


#include <DataFormats/EcalDigi/interface/EcalTriggerPrimitiveSample.h>
#include <DataFormats/EcalDigi/interface/EBDataFrame.h>
#include <DataFormats/EcalDigi/interface/EEDataFrame.h>

#include "FWCore/Framework/interface/EventSetup.h"

#include <vector> 
#include <iostream>

class EcalTPGFineGrainEBGroup;
class EcalTPGLutGroup;
class EcalTPGLutIdMap;
class EcalTPGFineGrainEBIdMap;
class EcalTPGFineGrainTowerEE;
class EcalTrigTowerDetId;
class EcalTPGTowerStatus;

/** 
    \class EcalFenixTcp
    \brief class representing the Fenix chip, format strip
*/
class EcalFenixTcp {

 private:
  bool debug_;

  int nbMaxStrips_;

  EcalFenixMaxof2 *maxOf2_;
  std::vector<EcalFenixBypassLin *> bypasslin_;
  EcalFenixEtTot *adder_;
  EcalFenixFgvbEB *fgvbEB_;
  EcalFenixTcpFgvbEE *fgvbEE_;
    
  EcalFenixTcpFormat *formatter_;

  // permanent data structures
  std::vector<std::vector<int> > bypasslin_out_;
  std::vector<int> adder_out_;
  std::vector<int> maxOf2_out_;
  std::vector<int> fgvb_out_;
   
 public:
  // temporary, for timing tests
  void setPointers(  const EcalTPGFineGrainEBGroup * ecaltpgFgEBGroup,
		     const EcalTPGLutGroup * ecaltpgLutGroup,
		     const EcalTPGLutIdMap * ecaltpgLut,
		     const EcalTPGFineGrainEBIdMap * ecaltpgFineGrainEB,
		     const EcalTPGFineGrainTowerEE * ecaltpgFineGrainTowerEE,
		     const EcalTPGTowerStatus * ecaltpgBadTT)
    {
      ecaltpgFgEBGroup_=ecaltpgFgEBGroup;
      ecaltpgLutGroup_=ecaltpgLutGroup;
      ecaltpgLut_=ecaltpgLut;
      ecaltpgFineGrainEB_=ecaltpgFineGrainEB;
      ecaltpgFineGrainTowerEE_=ecaltpgFineGrainTowerEE;
      ecaltpgBadTT_=ecaltpgBadTT;
   }
  // end temporary, for timing tests

  EcalFenixTcp(const edm::EventSetup & setup, bool tcpFormat, bool debug, bool famos, int binOfMax, int maxNrSamples,int nbMaxStrips);
  virtual ~EcalFenixTcp() ;

  void process(const edm::EventSetup & setup,
               std::vector <EBDataFrame> &bid,             //dummy argument for template call 
	       std::vector<std::vector<int> > & tpframetow, int nStr,
	       std::vector< EcalTriggerPrimitiveSample> & tptow,
	       std::vector< EcalTriggerPrimitiveSample> & tptow2,
	       bool isInInnerRings, EcalTrigTowerDetId thisTower);
  void process(const edm::EventSetup & setup,
               std::vector <EEDataFrame> &bid,             //dummy argument for template call 
	       std::vector<std::vector<int> > & tpframetow, int nStr,
	       std::vector< EcalTriggerPrimitiveSample> & tptow,
	       std::vector< EcalTriggerPrimitiveSample> & tptow2,
	       bool isInInnerRings, EcalTrigTowerDetId thisTower);

  void process_part1(std::vector<std::vector<int> > &tpframetow, int nStr,int bitMask);

  void  process_part2_barrel(std::vector<std::vector<int> > &, int nStr,
			     const EcalTPGFineGrainEBGroup *ecaltpgFgEBGroup,
			     const EcalTPGLutGroup*ecaltpgLutGroup,
			     const EcalTPGLutIdMap *ecaltpgLut,
			     const EcalTPGFineGrainEBIdMap *ecaltpgFineGrainEB,
			     const EcalTPGTowerStatus *ecaltpgBadTT,
			     std::vector< EcalTriggerPrimitiveSample> &tptow,
			     std::vector< EcalTriggerPrimitiveSample> &tptow2,
			     EcalTrigTowerDetId towid);
			       
  void  process_part2_endcap(std::vector<std::vector<int> > &, int nStr,int bitMask,
			     const EcalTPGLutGroup *ecaltpgLutGroup,
			     const EcalTPGLutIdMap *ecaltpgLut,
			     const EcalTPGFineGrainTowerEE *ecaltpgFineGrainTowerEE,
			     const EcalTPGTowerStatus *ecaltpgBadTT,
			     std::vector< EcalTriggerPrimitiveSample> &tptow,
			     std::vector< EcalTriggerPrimitiveSample> &tptow2,bool isInInnerRings,      
			     EcalTrigTowerDetId towid);
				   

  EcalFenixBypassLin *getBypasslin(int i) const {return bypasslin_[i];}
  EcalFenixEtTot *getAdder() const { return  adder_;}
  EcalFenixMaxof2 *getMaxOf2() const {return maxOf2_;}
  EcalFenixTcpFormat *getFormatter() const {return formatter_;}
  EcalFenixFgvbEB *getFGVBEB() const {return fgvbEB_;}
  EcalFenixTcpFgvbEE *getFGVBEE() const {return fgvbEE_;}
	        	      
  const EcalTPGFineGrainEBGroup * ecaltpgFgEBGroup_;
  const EcalTPGLutGroup * ecaltpgLutGroup_;
  const EcalTPGLutIdMap * ecaltpgLut_;
  const EcalTPGFineGrainEBIdMap * ecaltpgFineGrainEB_;
  const EcalTPGFineGrainTowerEE * ecaltpgFineGrainTowerEE_;
  const EcalTPGTowerStatus * ecaltpgBadTT_;
};


#endif
