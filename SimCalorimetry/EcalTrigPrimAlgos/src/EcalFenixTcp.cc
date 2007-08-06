#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixTcp.h>
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixTcpFormat.h>
#include <CondFormats/L1TObjects/interface/EcalTPParameters.h>
#include <iostream>
#include <vector>
//----------------------------------------------------------------------------------------
EcalFenixTcp::EcalFenixTcp(const EcalTPParameters * ecaltpp,bool tcpFormat, bool debug,bool famos, int binOfMax, int maxNrSamples): debug_(debug)
 { 
    bypasslin_.resize(EcalTPParameters::nbMaxStrips_);
    for (int i=0;i<EcalTPParameters::nbMaxStrips_;i++) bypasslin_[i] = new EcalFenixBypassLin();
    adder_= new EcalFenixEtTot();
    maxOf2_=new EcalFenixMaxof2(maxNrSamples);
    formatter_= new EcalFenixTcpFormat(ecaltpp, tcpFormat, debug_, famos, binOfMax);
    fgvbEB_= new EcalFenixFgvbEB(ecaltpp);
    fgvbEE_= new EcalFenixTcpFgvbEE(ecaltpp);

    // permanenet data structures
    bypasslin_out_.resize(EcalTPParameters::nbMaxStrips_);
    std::vector<int> vec(maxNrSamples,0);
    for (int i=0;i<EcalTPParameters::nbMaxStrips_;i++) bypasslin_out_[i]=vec;
    adder_out_.resize(maxNrSamples);
    maxOf2_out_.resize(maxNrSamples);
    fgvb_out_.resize(maxNrSamples);

  }
//-----------------------------------------------------------------------------------------  
  EcalFenixTcp::~EcalFenixTcp() {
    for (int i=0;i<EcalTPParameters::nbMaxStrips_;i++) delete bypasslin_[i];
    delete adder_; 
    delete maxOf2_;
    delete formatter_;
    delete fgvbEB_;
    delete fgvbEE_;
  }
//-----------------------------------------------------------------------------------------  
void EcalFenixTcp::process(std::vector <EBDataFrame> &,             //dummy argument for template call 
			   std::vector<std::vector<int> > & tpframetow, int nStr,
			   std::vector< EcalTriggerPrimitiveSample> & tptow,
			   std::vector< EcalTriggerPrimitiveSample> & tptow2,
			   int SM, int towerInSM)
{
	      
  int bitMask=12; //FIXME: to be verified
  process_part1(tpframetow,nStr,bitMask);
  process_part2_barrel(nStr,SM,towerInSM,tptow,tptow2);
}
 
//-----------------------------------------------------------------------------------------  
void EcalFenixTcp::process(std::vector <EEDataFrame> &,  //dummy argument for template call 
			   std::vector<std::vector<int> > & tpframetow, int nStr,
			   std::vector< EcalTriggerPrimitiveSample> & tptow,
			   std::vector< EcalTriggerPrimitiveSample> & tptow2,
			   int SM, int towerInSM)
{
	      
  int bitMask=10;
  process_part1(tpframetow,nStr,bitMask);
  process_part2_endcap(nStr,bitMask,SM,towerInSM,tptow,tptow2);
}
//----------------------------------------------------------------------------------------- 
void EcalFenixTcp::process_part1(std::vector<std::vector<int> > &tpframetow, int nStr, int bitMask)
{
 //call bypasslin
    for (int istrip=0;istrip<nStr;istrip ++){
      this->getBypasslin(istrip)->process(tpframetow[istrip],bypasslin_out_[istrip]);
    }
    //this is a test
    if (debug_) {
      std::cout<<"bypasslinout = "<<std::endl;
      for (int istrip=0;istrip<nStr;istrip ++){
	std::vector<int> stripin= bypasslin_out_[istrip];
	for (unsigned int is=0;is<stripin.size();is++){
	  std::cout<<stripin[is]<<" ";
	}
	std::cout<<std::endl;
      }
    }
    //call adder
    this->getAdder()->process(bypasslin_out_, nStr, bitMask,adder_out_);
    //this is a test:
    if (debug_) {
      std::cout<< "output of adder is a vector of size: "<<adder_out_.size()<<std::endl; 
      std::cout<< "value : "<<std::endl;
      for (unsigned int i =0; i<adder_out_.size();i++){
	std::cout <<" "<<adder_out_[i];
      }    
      std::cout<<std::endl;
    }
    //    return adder_out;
    return;
    
}
//-----------------------------------------------------------------------------------------
void EcalFenixTcp::process_part2_barrel(int nStr,int SM,int towerInSM,
					std::vector< EcalTriggerPrimitiveSample> & tcp_out,
                                        std::vector< EcalTriggerPrimitiveSample> & tcp_outTcc)
{
  //call maxof2
  this->getMaxOf2()->process(bypasslin_out_,nStr,maxOf2_out_);
  // this is a test:
  if (debug_) {
    std::cout<< "output of maxof2 is a vector of size: "<<maxOf2_out_.size()<<std::endl; 
    std::cout<< "value : "<<std::endl;
    for (unsigned int i =0; i<maxOf2_out_.size();i++){
      std::cout <<" "<<std::dec<<maxOf2_out_[i];
    }    
    std::cout<<std::endl;
  }
   
  //call fgvb
  this->getFGVBEB()->setParameters(SM, towerInSM);
  this->getFGVBEB()->process(adder_out_,maxOf2_out_,fgvb_out_);
  //this is a test:
  if (debug_) {
    std::cout<< "output of fgvb is a vector of size: "<<fgvb_out_.size()<<std::endl; 
    std::cout<< "value : "<<std::endl;
    for (unsigned int i =0; i<fgvb_out_.size();i++){
      std::cout <<" "<<std::dec<<fgvb_out_[i];
    }    
    std::cout<<std::endl;
  }

  // call formatter
  int eTTotShift=2;
  this->getFormatter()->setParameters(SM, towerInSM) ;
  this->getFormatter()->process(adder_out_,fgvb_out_,eTTotShift,tcp_out,tcp_outTcc);
  //this is a test:
  if (debug_) {
    std::cout<< "output of TCP formatter Barrel is a vector of size: "<<std::dec<<tcp_out.size()<<std::endl; 
    std::cout<< "value : "<<std::endl;
    for (unsigned int i =0; i<tcp_out.size();i++){
      std::cout <<" "<<i<<" "<<std::dec<<tcp_out[i];
    }    
    std::cout<<std::endl;
  }
    
  return;
}
//-----------------------------------------------------------------------------------------
void EcalFenixTcp::process_part2_endcap(int nStr, int bitMask,
                                        int SM,int towerInSM,
					std::vector< EcalTriggerPrimitiveSample> & tcp_out,
                                        std::vector< EcalTriggerPrimitiveSample> & tcp_outTcc)
{
  //call fgvb
  //  std::vector<int> fgvb_out;
  this->getFGVBEE()->setParameters(SM, towerInSM);
  //  fgvb_out = fgvbEE_->process(bypasslin_out,bitMask);
  fgvbEE_->process(bypasslin_out_,nStr,bitMask,fgvb_out_);

  //call formatter
  int eTTotShift=0;
  this->getFormatter()->setParameters(SM, towerInSM) ;
  this->getFormatter()->process(adder_out_,fgvb_out_,eTTotShift,tcp_out,tcp_outTcc);
  //this is a test:
  if (debug_) {
    std::cout<< "output of TCP formatter(endcap) is a vector of size: "<<std::dec<<tcp_out.size()<<std::endl; 
    std::cout<< "value : "<<std::endl;
    for (unsigned int i =0; i<tcp_out.size();i++){
      std::cout <<" "<<i<<" "<<std::dec<<tcp_out[i]<<std::endl;
    }    
    std::cout<<std::endl;
  }
  return;
}
