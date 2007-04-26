#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixTcp.h>
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixTcpFormat.h>
#include <iostream>
#include <vector>
//----------------------------------------------------------------------------------------
EcalFenixTcp::EcalFenixTcp(const EcalTPParameters * ecaltpp,bool tcpFormat, bool debug): debug_(debug)
 { 
    for (int i=0;i<nMaxStripsPerTower_;i++) bypasslin_[i] = new EcalFenixBypassLin();
    adder_= new EcalFenixEtTot();
    maxOf2_=new EcalFenixMaxof2();
    formatter_= new EcalFenixTcpFormat(ecaltpp,tcpFormat,debug_);
    fgvbEB_= new EcalFenixFgvbEB(ecaltpp);
    fgvbEE_= new EcalFenixTcpFgvbEE(ecaltpp);

  }
//-----------------------------------------------------------------------------------------  
  EcalFenixTcp::~EcalFenixTcp() {
    for (int i=0;i<nMaxStripsPerTower_;i++) delete bypasslin_[i];
    delete adder_; 
    delete maxOf2_;
    delete formatter_;
    delete fgvbEB_;
    delete fgvbEE_;
  }
//----------------------------------------------------------------------------------------- 
std::vector<int> EcalFenixTcp::process_part1(std::vector<std::vector<int> > tpframetow,std::vector<std::vector<int> >  & bypasslin_out,int bitMask)
{
 //call bypasslin
    for (unsigned int istrip=0;istrip<tpframetow.size();istrip ++){
      std::vector<int> stripin= tpframetow[istrip];
      bypasslin_out.push_back(this->getBypasslin(istrip)->process(stripin));
    }
    //this is a test
    if (debug_) {
      std::cout<<"bypasslinout = "<<std::endl;
      for (unsigned int istrip=0;istrip<bypasslin_out.size();istrip ++){
	std::vector<int> stripin= bypasslin_out[istrip];
	for (unsigned int is=0;is<stripin.size();is++){
	  std::cout<<stripin[is]<<" ";
	}
	std::cout<<std::endl;
      }
    }
    //call adder
    std::vector<int> adder_out;
    adder_out = this->getAdder()->process(bypasslin_out,bitMask);
    //this is a test:
    if (debug_) {
      std::cout<< "output of adder is a vector of size: "<<adder_out.size()<<std::endl; 
      std::cout<< "value : "<<std::endl;
      for (unsigned int i =0; i<adder_out.size();i++){
	std::cout <<" "<<adder_out[i];
      }    
      std::cout<<std::endl;
    }
    return adder_out;
    
}
//-----------------------------------------------------------------------------------------
void EcalFenixTcp::process_part2_barrel(std::vector<std::vector<int> > & bypasslin_out,
                                        std::vector<int> &adder_out,
                                        int SM,int towerInSM,
					std::vector< EcalTriggerPrimitiveSample> & tcp_out,
                                        std::vector< EcalTriggerPrimitiveSample> & tcp_outTcc)
{
  //call maxof2
  std::vector<int> maxof2_out;
  maxof2_out = this->getMaxOf2()->process(bypasslin_out);
  // this is a test:
  if (debug_) {
    std::cout<< "output of maxof2 is a vector of size: "<<adder_out.size()<<std::endl; 
    std::cout<< "value : "<<std::endl;
    for (unsigned int i =0; i<maxof2_out.size();i++){
      std::cout <<" "<<std::dec<<maxof2_out[i];
    }    
    std::cout<<std::endl;
  }
   
  //call fgvb
  std::vector<int> fgvb_out;
  this->getFGVBEB()->setParameters(SM, towerInSM);
  fgvb_out = this->getFGVBEB()->process(adder_out,maxof2_out);
  //this is a test:
  if (debug_) {
    std::cout<< "output of fgvb is a vector of size: "<<fgvb_out.size()<<std::endl; 
    std::cout<< "value : "<<std::endl;
    for (unsigned int i =0; i<fgvb_out.size();i++){
      std::cout <<" "<<std::dec<<fgvb_out[i];
    }    
    std::cout<<std::endl;
  }

  // call formatter
  int eTTotShift=2;
  this->getFormatter()->setParameters(SM, towerInSM) ;
  this->getFormatter()->process(adder_out,fgvb_out,eTTotShift,tcp_out,tcp_outTcc);
  //this is a test:
  if (debug_) {
    std::cout<< "output of formatter is a vector of size: "<<std::dec<<tcp_out.size()<<std::endl; 
    std::cout<< "value : "<<std::endl;
    for (unsigned int i =0; i<tcp_out.size();i++){
      std::cout <<" "<<i<<" "<<std::dec<<tcp_out[i];
    }    
    std::cout<<std::endl;
  }
    
  return;
}
//-----------------------------------------------------------------------------------------
void EcalFenixTcp::process_part2_endcap(std::vector<std::vector<int> > & bypasslin_out,
                                        std::vector<int> &adder_out,int bitMask,
                                        int SM,int towerInSM,
					std::vector< EcalTriggerPrimitiveSample> & tcp_out,
                                        std::vector< EcalTriggerPrimitiveSample> & tcp_outTcc)
{
  //call fgvb
  std::vector<int> fgvb_out;
  this->getFGVBEE()->setParameters(SM, towerInSM);
  fgvb_out = fgvbEE_->process(bypasslin_out,bitMask);

  //call formatter
  int eTTotShift=0;
  this->getFormatter()->setParameters(SM, towerInSM) ;
  this->getFormatter()->process(adder_out,fgvb_out,eTTotShift,tcp_out,tcp_outTcc);
  //this is a test:
  if (debug_) {
    std::cout<< "output of formatter is a vector of size: "<<std::dec<<tcp_out.size()<<std::endl; 
    std::cout<< "value : "<<std::endl;
    for (unsigned int i =0; i<tcp_out.size();i++){
      std::cout <<" "<<i<<" "<<std::dec<<tcp_out[i]<<std::endl;
    }    
    std::cout<<std::endl;
  }
  return;
}
