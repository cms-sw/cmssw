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
#include <vector> 
#include <iostream>

class EcalTPParameters;

/** 
    \class EcalFenixTcp
    \brief class representing the Fenix chip, format strip
*/
class EcalFenixTcp {

 private:

  EcalFenixMaxof2 *maxOf2_;
  std::vector<EcalFenixBypassLin *> bypasslin_;
  EcalFenixEtTot *adder_;
  EcalFenixFgvbEB *fgvbEB_;
  EcalFenixTcpFgvbEE *fgvbEE_;
    
  EcalFenixTcpFormat *formatter_;
 
  bool debug_;

  // permanent data structures
  std::vector<std::vector<int> > bypasslin_out_;
  std::vector<int> adder_out_;
  std::vector<int> maxOf2_out_;
  std::vector<int> fgvb_out_;
   
 public:
  EcalFenixTcp(const EcalTPParameters *,bool tcpFormat, bool debug, bool famos, int binOfMax, int maxNrSamples);
  virtual ~EcalFenixTcp() ;

  void process(std::vector <const EBDataFrame *> &bid,             //dummy argument for template call 
	       std::vector<std::vector<int> > & tpframetow, int nStr,
	       std::vector< EcalTriggerPrimitiveSample> & tptow,
	       std::vector< EcalTriggerPrimitiveSample> & tptow2,
	       int SM, int towerInSM);
  void process(std::vector <const EEDataFrame *> &bid,             //dummy argument for template call 
	       std::vector<std::vector<int> > & tpframetow, int nStr,
	       std::vector< EcalTriggerPrimitiveSample> & tptow,
	       std::vector< EcalTriggerPrimitiveSample> & tptow2,
	       int SM, int towerInSM);

  void process_part1(std::vector<std::vector<int> > &tpframetow, int nStr,int bitMask);
  void  process_part2_barrel(int nStr,int SM,int towerInSM,
			     std::vector< EcalTriggerPrimitiveSample> &tptow,
			     std::vector< EcalTriggerPrimitiveSample> &tptow2);
			       
  void  process_part2_endcap(int nStr,int bitMask,int SM,int towerInSM,
			     std::vector< EcalTriggerPrimitiveSample> &tptow,
			     std::vector< EcalTriggerPrimitiveSample> &tptow2);
				   

  EcalFenixBypassLin *getBypasslin(int i) const {return bypasslin_[i];}
  EcalFenixEtTot *getAdder() const { return  adder_;}
  EcalFenixMaxof2 *getMaxOf2() const {return maxOf2_;}
  EcalFenixTcpFormat *getFormatter() const {return formatter_;}
  EcalFenixFgvbEB *getFGVBEB() const {return fgvbEB_;}
  EcalFenixTcpFgvbEE *getFGVBEE() const {return fgvbEE_;}
	        	      
};


#endif
