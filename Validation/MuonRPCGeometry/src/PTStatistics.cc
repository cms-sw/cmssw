#include "Validation/MuonRPCGeometry/interface/PTStatistics.h"

#include <sstream>
#include <numeric>
#include <iostream>
//-----------------------------------------------------------------------------
//  PTStatistics
//-----------------------------------------------------------------------------
static std::vector<long double> initializeRates()
{
  std::vector<long double> rates(RPCpg::ptBins_s,0);
  //std::cout << "Initilizing rates" << std::endl;
  
  // Note bin=0 is empty during generation
  // bin=0 is used only when calculating efficiencies (for storing muons,that werent found)
  for (unsigned int i = 1;i < rates.size(); ++i ){
    
    long double low =  RPCpg::pts[i];
    long double high =  RPCpg::pts[i+1];
    long double rt = RPCpg::rate(low)-RPCpg::rate(high);

    /* std::cout << "PtCode " << i
              << " " << low
              << " " << high
              << " " << rt
              << std::endl;*/
    rates[i] = rt;
  }
  return rates;
}

const std::vector<long double> PTStatistics::m_rates = initializeRates();

PTStatistics::PTStatistics(){
   this->assign(RPCpg::ptBins_s,0);
}   


void PTStatistics::update(PTStatistics & otherPtStats){
   
   for (unsigned int i=0; i<this->size();++i){
      //this->at(i)+=otherPtStats.at(i);
      (*this)[i]+=otherPtStats[i];
   }
   

}
std::string PTStatistics::toString(){
   
   std::stringstream ss;
   ss << "PTStats:";
   for (unsigned int i=0; i<this->size();++i){
      ss << " " << (*this)[i];
   }
   
   return ss.str();
}

long double PTStatistics::eff(int ptCut){  // ptCut=0 -> total rate
   //int eqOrAbovePtCut = 0;
   //for(unsigned int i=ptCut;i<this->size();++i) eqOrAbovePtCut += this->at(i);
   // return double(eqOrAbovePtCut)/this->sum();
   return double(sum(ptCut))/this->sum();
}


long int PTStatistics::sum(const int & ptCut) const{
//inline int PTStatistics::sum(const int & ptCut) const{  
   //return std::accumulate(this->begin(),this->end(),0);
   long int eqOrAbovePtCut = 0;
   unsigned int size = this->size();
   //for(unsigned int i=ptCut;i<size;++i) eqOrAbovePtCut += this->at(i);
   for(unsigned int i=ptCut;i<size;++i) eqOrAbovePtCut += (*this)[i];
   return eqOrAbovePtCut;
}

//inline int PTStatistics::sum() const{
long int PTStatistics::sum() const{  
   //return std::accumulate(this->begin(),this->end(),0);
   long int eqOrAbovePtCut = 0;
   //unsigned int size = this->size();
   //for(unsigned int i=ptCut;i<size;++i) eqOrAbovePtCut += this->at(i);
   //for(unsigned int i=ptCut;i<size;++i) eqOrAbovePtCut += (*this)[i];
   PTStatistics::const_iterator it = this->begin();
   PTStatistics::const_iterator itend = this->end();
   for(;it!=itend;++it) eqOrAbovePtCut += *it;
   
   return eqOrAbovePtCut;
}

long double PTStatistics::sumR(const int & ptCut) const{ 
   //return std::accumulate(this->begin(),this->end(),0);
   long double eqOrAbovePtCut = 0;
   unsigned int size = this->size();
   //for(unsigned int i=ptCut;i<size;++i) eqOrAbovePtCut += this->at(i);
   for(unsigned int i=ptCut;i<size;++i) eqOrAbovePtCut += (*this)[i]*m_rates[i];
   return eqOrAbovePtCut;
}

long double PTStatistics::sumR() const{ 
   //return std::accumulate(this->begin(),this->end(),0);
   long double eqOrAbovePtCut = 0;
   //unsigned int size = this->size();
   //for(unsigned int i=ptCut;i<size;++i) eqOrAbovePtCut += this->at(i);
   //for(unsigned int i=ptCut;i<size;++i) eqOrAbovePtCut += (*this)[i];
   //PTStatistics::const_iterator it = this->begin();
   //PTStatistics::const_iterator itend = this->end();
   //PtCode 0 - muons not found
   unsigned int size = this->size();
   for(unsigned int i=1;i<size;++i) eqOrAbovePtCut += (*this)[i]*m_rates[i];
   //for(;it!=itend;++it) eqOrAbovePtCut += *it*m_rates[i];
   
   return eqOrAbovePtCut;
}

