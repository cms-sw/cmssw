#include <SimCalorimetry/EcalEBTrigPrimAlgos/interface/EcalEBPhase2TPFormatter.h>
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>

EcalEBPhase2TPFormatter::EcalEBPhase2TPFormatter(bool debug)  
  : debug_(debug)
{}

EcalEBPhase2TPFormatter::~EcalEBPhase2TPFormatter() {}



void EcalEBPhase2TPFormatter::process(std::vector<int> & amp,
				 std::vector<int64_t> & time,
				 std::vector<int> &outEt,
                                 std::vector<int64_t> &outTime) {

  unsigned int size=amp.size();
  outEt.resize(size);
  outTime.resize(size);
  
  for (unsigned int i = 0; i<size; ++i ) {
    outEt[i]  = amp[i];
    outTime[i] = time[i];
  }


  for (unsigned int i = 0; i<size; ++i ) {
    // this is the energy compression to 12 bits to go in the DF. To be done as last thing before building the TP    
    //Bit shift by 1 to go from 13 bits to 12                                                                                         
    outEt[i] = outEt[i] >> 1 ;
    if (outEt[i] > 0xFFF ) outEt[i] = 0xFFF;


  }


  for (unsigned int i = 0; i<size; ++i ) {
    // this is the time compression to 5 bits to go in the DF. 
    outTime [i]= outTime [i] >> 6 ;
    if ( outTime[i] > 0xf ) outTime[i] = 0xf;
    else if  ( outTime[i] <  -0x10 ) outTime[i] = -0x10; 




  }



}


