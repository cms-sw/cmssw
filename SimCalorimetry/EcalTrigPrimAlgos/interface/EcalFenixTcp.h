#ifndef ECAL_FENIX_TCP_H
#define ECAL_FENIX_TCP_H

#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixChip.h>
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
//class EcalFenixTcp : public EcalFenixChip {
  
class EcalFenixTcp {

 private:

  EcalFenixMaxof2 *maxOf2_;
  EcalFenixBypassLin *bypasslin_[nMaxStripsPerTower_];
  EcalFenixEtTot *adder_;
  //ENDCAP:MODIF
  EcalFenixFgvbEB *fgvbEB_;
  EcalFenixTcpFgvbEE *fgvbEE_;
    
  EcalFenixTcpFormat *formatter_;
 
  bool debug_;
    
 public:
  EcalFenixTcp(const EcalTPParameters *,bool tcpFormat, bool debug);
  virtual ~EcalFenixTcp() ;

  //void process(std::vector<std::vector<int> > & tpframetow,std::vector< EcalTriggerPrimitiveSample> & tptow,
  //	 int SM, int towerInSM);
  //    template <class T> void process(std::vector<const T *> &,std::vector<std::vector<int> > & tpframetow,
  //                                    std::vector< EcalTriggerPrimitiveSample> & tptow,
  //                                     std::vector< EcalTriggerPrimitiveSample> & tptow2,
  //                             	    int SM, int towerInSM);


  std::vector<int> process_part1(std::vector<std::vector<int> > tpframetow,
				 std::vector<std::vector<int> > &bypasslin_out,int bitMask);
  void  process_part2_barrel(std::vector<std::vector<int> > & bypasslin_out,
			     std::vector<int> &adder_out,
			     int SM,int towerInSM,
			     std::vector< EcalTriggerPrimitiveSample> &tptow,
			     std::vector< EcalTriggerPrimitiveSample> &tptow2);
			       
  void  process_part2_endcap(std::vector<std::vector<int> > & bypasslin_out,
			     std::vector<int> &adder_out,
			     int bitMask,int SM,int towerInSM,
			     std::vector< EcalTriggerPrimitiveSample> &tptow,
			     std::vector< EcalTriggerPrimitiveSample> &tptow2);
				   

  EcalFenixBypassLin *getBypasslin(int i) const {return bypasslin_[i];}
  //    EcalFenixEtTot *getAdder() const { return  dynamic_cast<EcalFenixEtTot *>(adder_);}
  EcalFenixEtTot *getAdder() const { return  adder_;}
  EcalFenixMaxof2 *getMaxOf2() const {return maxOf2_;}
  //ENDCAP:MODIF
  //EcalFenixTcpFormat *getFormatter() const {return dynamic_cast<EcalFenixTcpFormat *>( formatter_);}
  EcalFenixTcpFormat *getFormatter() const {return formatter_;}
  //ENDCAP:MODIF
  //EcalFenixFgvbEB *getFGVB() const {return dynamic_cast<EcalFenixFgvbEB *>(fgvb_);}
  EcalFenixFgvbEB *getFGVBEB() const {return fgvbEB_;}
  EcalFenixTcpFgvbEE *getFGVBEE() const {return fgvbEE_;}
     
   
    
  // ========================= implementations ==========================================    
  void process(std::vector <const EBDataFrame *> &bid,
	       std::vector<std::vector<int> > & tpframetow,
	       std::vector< EcalTriggerPrimitiveSample> & tptow,
	       std::vector< EcalTriggerPrimitiveSample> & tptow2,
	       int SM, int towerInSM)
    {
	      
      int bitMask=12; //FIXME: to be verified
      std::vector<std::vector<int> > bypasslin_out;	        
      std::vector<int> adder_out=process_part1(tpframetow,bypasslin_out,bitMask);
      process_part2_barrel(bypasslin_out,adder_out,SM,towerInSM,tptow,tptow2);
    }
 
 
 
  void process(std::vector <const EEDataFrame *> &bid,
	       std::vector<std::vector<int> > & tpframetow,
	       std::vector< EcalTriggerPrimitiveSample> & tptow,
	       std::vector< EcalTriggerPrimitiveSample> & tptow2,
	       int SM, int towerInSM)
    {
	      
      int bitMask=10;
      std::vector<std::vector<int> > bypasslin_out;	        
     
      std::vector<int> adder_out=process_part1(tpframetow,bypasslin_out,bitMask);
      process_part2_endcap(bypasslin_out,adder_out,bitMask,SM,towerInSM,tptow,tptow2);
    }
 	        	      
};


#endif
