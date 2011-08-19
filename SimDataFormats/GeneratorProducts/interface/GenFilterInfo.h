#ifndef SimDataFormats_GeneratorProducts_GenFilterInfo_h
#define SimDataFormats_GeneratorProducts_GenFilterInfo_h

/** \class GenFilterInfo
 *
 */

#include <cmath>

class GenFilterInfo {
public:
  
  // constructors, destructors
  GenFilterInfo();
  GenFilterInfo(unsigned int, unsigned int);
  virtual ~GenFilterInfo();
  
  // getters
  unsigned int numEventsTried() const { return numEventsTried_;}
  unsigned int numEventsPassed() const { return numEventsPassed_;}
  double filterEfficiency() const { return ( numEventsTried_ > 0 ? (double)numEventsPassed_/(double)numEventsTried_ : 1. ) ; }
  double filterEfficiencyError() const { return ( numEventsTried_ > 0 ? std::sqrt((double)numEventsPassed_*(1.-(double)numEventsPassed_/(double)numEventsTried_))/(double)numEventsTried_ : 1. ); }

  // merge method. It will be used when merging different jobs populating the same lumi section
  bool mergeProduct(GenFilterInfo const &other);
  
  
private:
  unsigned int  numEventsTried_;
  unsigned int  numEventsPassed_;

};


#endif // SimDataFormats_GeneratorProducts_GenFilterInfo_h
