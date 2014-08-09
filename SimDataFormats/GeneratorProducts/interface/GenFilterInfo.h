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
  GenFilterInfo(unsigned int, unsigned int, double, double, double, double, double, double, double, double);
  GenFilterInfo(const GenFilterInfo&);
  virtual ~GenFilterInfo();
  
  // getters
  unsigned int numEventsTried() const { return numEventsTried_;}
  unsigned int numEventsPassed() const { return numEventsPassed_;}

  double sumPositiveWeights() const {return (sumPassPositiveWeights_+sumFailPositiveWeights_);}
  double sumNegativeWeights() const {return (sumPassNegativeWeights_+sumFailNegativeWeights_);}

  double sumPassWeights() const {return (sumPassPositiveWeights_+sumPassNegativeWeights_);}

  double sumFailWeights() const {return (sumFailPositiveWeights_+sumFailNegativeWeights_);}

  double sumWeights() const {return (sumPassWeights() + sumFailWeights());}

  double sumPassPositiveWeights() const {return sumPassPositiveWeights_;}
  double sumPassPositiveWeights2() const {return sumPassPositiveWeights2_;}
  double sumPassNegativeWeights() const {return sumPassNegativeWeights_;}
  double sumPassNegativeWeights2() const {return sumPassNegativeWeights2_;}
	                                                       
  double sumFailPositiveWeights() const {return sumFailPositiveWeights_;}
  double sumFailPositiveWeights2() const {return sumFailPositiveWeights2_;}
  double sumFailNegativeWeights()	const {return sumFailNegativeWeights_;}
  double sumFailNegativeWeights2() const {return sumFailNegativeWeights2_;}


  double filterEfficiency() const { return ( numEventsTried_ > 0 ? (double)numEventsPassed_/(double)numEventsTried_ : 1. ) ; }
  double filterEfficiencyError() const { return ( numEventsTried_ > 0 ? std::sqrt((double)numEventsPassed_*(1.-(double)numEventsPassed_/(double)numEventsTried_))/(double)numEventsTried_ : 1. ); }
  
  double filterWeightedEfficiency() const;
  double filterWeightedEfficiencyError() const;
  double filterWeightedEfficiencyErrorBinomial() const; 

  // merge method. It will be used when merging different jobs populating the same lumi section
  bool mergeProduct(GenFilterInfo const &other);
  
  
private:
  unsigned int  numEventsTried_;
  unsigned int  numEventsPassed_;
  double        sumPassPositiveWeights_;
  double        sumPassPositiveWeights2_;
  double        sumPassNegativeWeights_;
  double        sumPassNegativeWeights2_;

  double        sumFailPositiveWeights_;
  double        sumFailPositiveWeights2_;
  double        sumFailNegativeWeights_;
  double        sumFailNegativeWeights2_;

};


#endif // SimDataFormats_GeneratorProducts_GenFilterInfo_h
