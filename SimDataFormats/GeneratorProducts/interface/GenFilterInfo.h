#ifndef SimDataFormats_GeneratorProducts_GenFilterInfo_h
#define SimDataFormats_GeneratorProducts_GenFilterInfo_h

/** \class GenFilterInfo
 *
 */

#include <cmath>
#include <iostream>

class GenFilterInfo {
public:
  
  // constructors, destructors
  GenFilterInfo();
  GenFilterInfo(unsigned int, unsigned int); // obsolete, should be avoided for new classes
  GenFilterInfo(unsigned int, unsigned int, unsigned int, unsigned int,
		double, double, double, double);
  GenFilterInfo(const GenFilterInfo&);
  virtual ~GenFilterInfo();
  
  // getters
  unsigned int numEventsTried() const { return (numTotalPositiveEvents_ + numTotalNegativeEvents_);}
  unsigned int numEventsPassed() const { return fmax(0, (numPassPositiveEvents_ - numPassNegativeEvents_));}
  unsigned int numEventsTotal() const { return fmax(0, (numTotalPositiveEvents_ - numTotalNegativeEvents_));}

  unsigned int  numPassPositiveEvents() const { return numPassPositiveEvents_;}
  unsigned int  numTotalPositiveEvents() const { return numTotalPositiveEvents_;}

  unsigned int  numPassNegativeEvents() const { return numPassNegativeEvents_;}
  unsigned int  numTotalNegativeEvents() const { return numTotalNegativeEvents_;}


  double sumPassWeights() const { return sumPassWeights_;}
  double sumPassWeights2() const { return sumPassWeights2_;}

  double sumFailWeights() const { return sumTotalWeights_ - sumPassWeights_;}
  double sumFailWeights2() const { return sumTotalWeights2_ - sumPassWeights2_;}

  double sumWeights() const { return sumTotalWeights_;}
  double sumWeights2() const { return sumTotalWeights2_;}

  double filterEfficiency(int idwtup=+3) const;
  double filterEfficiencyError(int idwtup=+3) const;
  // merge method. It will be used when merging different jobs populating the same lumi section
  bool mergeProduct(GenFilterInfo const &other);
  void swap(GenFilterInfo& other);

private:
  
  unsigned int  numPassPositiveEvents_;
  unsigned int  numPassNegativeEvents_;
  unsigned int  numTotalPositiveEvents_;
  unsigned int  numTotalNegativeEvents_;

  double        sumPassWeights_;
  double        sumPassWeights2_;
  double        sumTotalWeights_;
  double        sumTotalWeights2_;



};


#endif // SimDataFormats_GeneratorProducts_GenFilterInfo_h
