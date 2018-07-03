#ifndef TkNavigation_TkMSParameterization_H
#define TkNavigation_TkMSParameterization_H

#include "TrackingTools/DetLayers/interface/DetLayer.h"

#include<array>
#include<vector>
#include<unordered_map>
#include<cmath>
#include<algorithm>

#include<ostream>


class TkMSParameterizationBuilder;
namespace tkMSParameterization {

  constexpr unsigned short packLID(unsigned int id, unsigned int od) {  return (id<<8) | od ;}
  constexpr std::tuple<unsigned short, unsigned short> unpackLID(unsigned short lid) { return std::make_tuple(lid>>8, lid&255);}



  constexpr unsigned int nLmBins() { return 12*10;}                                                                         
  constexpr float lmBin() { return 0.1f;}
  constexpr float lmBinInv() { return 1.f/lmBin();}

  struct Elem {
    float vi;
    float vo;
    float uerr;
    float verr;
  };

  // this shall be sorted by "vo"
  class Elems {
  public:
    Elem find(float v) {
      auto p = find_if(data.begin(),data.end(),[=](Elem const & d) { return d.vo>v;});
      if (p!=data.begin()) --p; 
      return *p;
    }
   auto const& operator()() const { return data;}

  private:
    std::vector<Elem> data;
    friend TkMSParameterizationBuilder;
  }; 

  class FromToData  {
  public:

     Elems const & get(float tnLambda) const {
       auto i = std::min(nLmBins()-1,(unsigned int)(std::abs(tnLambda)*lmBinInv()));
       return data[i];
     }

    auto const& operator()() const { return data;}
  private:
     std::array<Elems,nLmBins()> data;
     friend TkMSParameterizationBuilder;
  };
  
  using AllData = std::unordered_map<unsigned short, FromToData>;
  
}

inline
std::ostream & operator<<(std::ostream & os, tkMSParameterization::Elem d) {
  os <<d.vi<<'/'<<d.vo<<':'<<d.uerr<<'/'<<d.verr;
  return os;
}


class TkMSParameterization {
public: 
  using FromToData = tkMSParameterization::FromToData;
  using	AllData = tkMSParameterization::AllData;

  FromToData const * fromTo(DetLayer const & in, DetLayer const & out) const {
   return fromTo(in.seqNum(),out.seqNum());
  }


 FromToData const * fromTo(int in, int out) const {
    using namespace tkMSParameterization;
    auto id = packLID(in,out);   
    auto p = data.find(id);
    if (p!=data.end()) return &(*p).second;
    return nullptr;
  }
 

  auto const& operator()() const { return data;}
 
private:
  AllData data;
  friend TkMSParameterizationBuilder;
};



#endif // TkNavigation_TkMSParameterization_H

