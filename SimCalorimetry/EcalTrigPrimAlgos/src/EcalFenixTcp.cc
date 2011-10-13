#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixTcp.h>
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixTcpFormat.h>

#include "CondFormats/EcalObjects/interface/EcalTPGFineGrainEBGroup.h"
#include "CondFormats/EcalObjects/interface/EcalTPGLutGroup.h"
#include "CondFormats/EcalObjects/interface/EcalTPGLutIdMap.h"
#include "CondFormats/EcalObjects/interface/EcalTPGFineGrainEBIdMap.h"
#include "CondFormats/EcalObjects/interface/EcalTPGFineGrainTowerEE.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include <vector>
//----------------------------------------------------------------------------------------
EcalFenixTcp::EcalFenixTcp(const edm::EventSetup& setup,bool tcpFormat, bool debug,bool famos, int binOfMax, int maxNrSamples, int nbMaxStrips): debug_(debug),nbMaxStrips_(nbMaxStrips)
 { 
    bypasslin_.resize(nbMaxStrips_);
    for (int i=0;i<nbMaxStrips_;i++) bypasslin_[i] = new EcalFenixBypassLin();
    adder_= new EcalFenixEtTot();
    maxOf2_=new EcalFenixMaxof2(maxNrSamples,nbMaxStrips_);
    formatter_= new EcalFenixTcpFormat(tcpFormat, debug_, famos, binOfMax);
    fgvbEB_= new EcalFenixFgvbEB(maxNrSamples);
    fgvbEE_= new EcalFenixTcpFgvbEE(maxNrSamples);

    // permanent data structures
    bypasslin_out_.resize(nbMaxStrips_);
    std::vector<int> vec(maxNrSamples,0);
    for (int i=0;i<nbMaxStrips_;i++) bypasslin_out_[i]=vec;
    adder_out_.resize(maxNrSamples);
    maxOf2_out_.resize(maxNrSamples);
    fgvb_out_.resize(maxNrSamples);

  }
//-----------------------------------------------------------------------------------------  
  EcalFenixTcp::~EcalFenixTcp() {
    for (int i=0;i<nbMaxStrips_;i++) delete bypasslin_[i];
    delete adder_; 
    delete maxOf2_;
    delete formatter_;
    delete fgvbEB_;
    delete fgvbEE_;
  }
//-----------------------------------------------------------------------------------------  

void EcalFenixTcp::process(const edm::EventSetup& setup,
                           std::vector <EBDataFrame> &bid,             //dummy argument for template call 
			   std::vector<std::vector<int> > & tpframetow, int nStr,
			   std::vector< EcalTriggerPrimitiveSample> & tptow,
			   std::vector< EcalTriggerPrimitiveSample> & tptow2,
			   bool isInInnerRing, EcalTrigTowerDetId towid) 
{

  int bitMask=12; 
  process_part1(tpframetow,nStr,bitMask);

 
process_part2_barrel(tpframetow,nStr,ecaltpgFgEBGroup_,ecaltpgLutGroup_,ecaltpgLut_,ecaltpgFineGrainEB_,ecaltpgBadTT_,tptow,tptow2,towid);
}
 
//-----------------------------------------------------------------------------------------  
void EcalFenixTcp::process(const edm::EventSetup& setup,
                           std::vector <EEDataFrame> &bid,  //dummy argument for template call 
			   std::vector<std::vector<int> > & tpframetow, int nStr,
			   std::vector< EcalTriggerPrimitiveSample> & tptow,
			   std::vector< EcalTriggerPrimitiveSample> & tptow2,
			   bool isInInnerRing, EcalTrigTowerDetId towid) 
{
  int bitMask=12; // Pascal: endcap has 12 bits as in EB (bug in FENIX!!!!) {was 10 before]
  process_part1(tpframetow,nStr,bitMask);
 
  process_part2_endcap(tpframetow,nStr,bitMask,ecaltpgLutGroup_,ecaltpgLut_,ecaltpgFineGrainTowerEE_,ecaltpgBadTT_,tptow,tptow2,isInInnerRing, towid);
}
//----------------------------------------------------------------------------------------- 
void EcalFenixTcp::process_part1(std::vector<std::vector<int> > &tpframetow, int nStr, int bitMask)
{
//  //call bypasslin
//     for (int istrip=0;istrip<nStr;istrip ++){
//       this->getBypasslin(istrip)->process(tpframetow[istrip],bypasslin_out_[istrip]);
//     }
//     //this is a test
//     if (debug_) {
//       std::cout<<"bypasslinout = "<<std::endl;
//       for (int istrip=0;istrip<nStr;istrip ++){
// 	std::vector<int> stripin= bypasslin_out_[istrip];
// 	for (unsigned int is=0;is<stripin.size();is++){
// 	  std::cout<<stripin[is]<<" ";
// 	}
// 	std::cout<<std::endl;
//       }
//     }

//     //call adder
//     this->getAdder()->process(bypasslin_out_, nStr, bitMask,adder_out_);
     this->getAdder()->process(tpframetow, nStr, bitMask,adder_out_);
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
void EcalFenixTcp::process_part2_barrel(std::vector<std::vector<int> > & bypasslinout, int nStr,
                                        const EcalTPGFineGrainEBGroup *ecaltpgFgEBGroup,
                                        const EcalTPGLutGroup *ecaltpgLutGroup,
                                        const EcalTPGLutIdMap *ecaltpgLut,
                                        const EcalTPGFineGrainEBIdMap *ecaltpgFineGrainEB,
					const EcalTPGTowerStatus *ecaltpgBadTT,
					std::vector< EcalTriggerPrimitiveSample> & tcp_out,
                                        std::vector< EcalTriggerPrimitiveSample> & tcp_outTcc,
					EcalTrigTowerDetId towid)
{
  //call maxof2
  //  this->getMaxOf2()->process(bypasslin_out_,nStr,maxOf2_out_);
  this->getMaxOf2()->process(bypasslinout,nStr,maxOf2_out_);
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

  this->getFGVBEB()->setParameters(towid.rawId(),ecaltpgFgEBGroup,ecaltpgFineGrainEB);
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
 
  this->getFormatter()->setParameters(towid.rawId(),ecaltpgLutGroup,ecaltpgLut,ecaltpgBadTT);
  this->getFormatter()->process(adder_out_,fgvb_out_,eTTotShift,tcp_out,tcp_outTcc,false);
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
void EcalFenixTcp::process_part2_endcap(std::vector<std::vector<int> > & bypasslinout, int nStr, int bitMask,
                                        const EcalTPGLutGroup *ecaltpgLutGroup,
                                        const EcalTPGLutIdMap *ecaltpgLut,
                                        const EcalTPGFineGrainTowerEE *ecaltpgFineGrainTowerEE,
					const EcalTPGTowerStatus *ecaltpgbadTT,
					std::vector< EcalTriggerPrimitiveSample> & tcp_out,
                                        std::vector< EcalTriggerPrimitiveSample> & tcp_outTcc,bool isInInnerRings,					
					EcalTrigTowerDetId towid)

{
  //call fgvb
  this->getFGVBEE()->setParameters(towid.rawId(),ecaltpgFineGrainTowerEE);
  //  fgvbEE_->process(bypasslin_out_,nStr,bitMask,fgvb_out_);
  fgvbEE_->process(bypasslinout,nStr,bitMask,fgvb_out_);

  //call formatter
  int eTTotShift=2; // Pascal: endcap has 12 bits as in EB (bug in FENIX!!!!) so shift must be applied to just keep [11:2]

  this->getFormatter()->setParameters(towid.rawId(),ecaltpgLutGroup,ecaltpgLut,ecaltpgbadTT);

  this->getFormatter()->process(adder_out_,fgvb_out_,eTTotShift,tcp_out,tcp_outTcc,isInInnerRings);
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
