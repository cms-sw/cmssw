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
  unsigned int numEventsTried() const { return (numPassPositiveEvents_ + numPassNegativeEvents_ + numFailPositiveEvents_ + numFailNegativeEvents_);}
  unsigned int numEventsPassed() const { return fmax(0, (numPassPositiveEvents_ - numPassNegativeEvents_));}
  unsigned int numEventsTotal() const { return fmax(0, (numPassPositiveEvents_ + numFailPositiveEvents_ - numPassNegativeEvents_ - numFailNegativeEvents_));}

  unsigned int  numPassPositiveEvents() const { return numPassPositiveEvents_;}
  unsigned int  numFailPositiveEvents() const { return numFailPositiveEvents_;}
  unsigned int  numPositiveEvents() const { return (numPassPositiveEvents_+numFailPositiveEvents_);}

  unsigned int  numPassNegativeEvents() const { return numPassNegativeEvents_;}
  unsigned int  numFailNegativeEvents() const { return numFailNegativeEvents_;}
  unsigned int  numNegativeEvents() const { return (numPassNegativeEvents_+numFailNegativeEvents_);}


  double sumPassWeights() const { return sumPassWeights_;}
  double sumPassWeights2() const { return sumPassWeights2_;}

  double sumFailWeights() const { return sumFailWeights_;}
  double sumFailWeights2() const { return sumFailWeights2_;}

  double sumWeights() const { return (sumPassWeights() + sumFailWeights());}
  double sumWeights2() const { return (sumPassWeights2() + sumFailWeights2());}

  double filterEfficiency(int idwtup=+3) const;
  double filterEfficiencyError(int idwtup=+3) const;
  // merge method. It will be used when merging different jobs populating the same lumi section
  bool mergeProduct(GenFilterInfo const &other);
  
private:
  
  unsigned int  numPassPositiveEvents_;
  unsigned int  numPassNegativeEvents_;
  unsigned int  numFailPositiveEvents_;
  unsigned int  numFailNegativeEvents_;

  double        sumPassWeights_;
  double        sumPassWeights2_;
  double        sumFailWeights_;
  double        sumFailWeights2_;



};


#endif // SimDataFormats_GeneratorProducts_GenFilterInfo_h
