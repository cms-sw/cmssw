#include <iostream>
#include <algorithm> 

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "SimDataFormats/GeneratorProducts/interface/GenFilterInfo.h"

using namespace edm;
using namespace std;

GenFilterInfo::GenFilterInfo() :
  numEventsTried_(0),
  numEventsPassed_(0),
  sumPassPositiveWeights_(0.),
  sumPassPositiveWeights2_(0.),
  sumPassNegativeWeights_(0.),
  sumPassNegativeWeights2_(0.),
  sumFailPositiveWeights_(0.),
  sumFailPositiveWeights2_(0.),
  sumFailNegativeWeights_(0.),
  sumFailNegativeWeights2_(0.)
{
}

GenFilterInfo::GenFilterInfo(unsigned int tried_, unsigned int passed_) :
  numEventsTried_(tried_),
  numEventsPassed_(passed_),
  sumPassPositiveWeights_(0.),
  sumPassPositiveWeights2_(0.),
  sumPassNegativeWeights_(0.),
  sumPassNegativeWeights2_(0.),
  sumFailPositiveWeights_(0.),
  sumFailPositiveWeights2_(0.),
  sumFailNegativeWeights_(0.),
  sumFailNegativeWeights2_(0.)
{
}

GenFilterInfo::GenFilterInfo(unsigned int tried_, unsigned int passed_,
			     double passpw_, double passpw2_,
			     double passnw_, double passnw2_,
			     double failpw_, double failpw2_,
			     double failnw_, double failnw2_) :
  numEventsTried_(tried_),
  numEventsPassed_(passed_),
  sumPassPositiveWeights_(passpw_),
  sumPassPositiveWeights2_(passpw2_),
  sumPassNegativeWeights_(passnw_),
  sumPassNegativeWeights2_(passnw2_),
  sumFailPositiveWeights_(failpw_),
  sumFailPositiveWeights2_(failpw2_),
  sumFailNegativeWeights_(failnw_),
  sumFailNegativeWeights2_(failnw2_)
{
}

GenFilterInfo::GenFilterInfo(const GenFilterInfo& other):
  numEventsTried_(other.numEventsTried_),
  numEventsPassed_(other.numEventsPassed_),
  sumPassPositiveWeights_(other.sumPassPositiveWeights_),
  sumPassPositiveWeights2_(other.sumPassPositiveWeights2_),
  sumPassNegativeWeights_(other.sumPassNegativeWeights_),
  sumPassNegativeWeights2_(other.sumPassNegativeWeights2_),
  sumFailPositiveWeights_(other.sumFailPositiveWeights_),
  sumFailPositiveWeights2_(other.sumFailPositiveWeights2_),
  sumFailNegativeWeights_(other.sumFailNegativeWeights_),
  sumFailNegativeWeights2_(other.sumFailNegativeWeights2_)
{
}

GenFilterInfo::~GenFilterInfo()
{
}

bool GenFilterInfo::mergeProduct(GenFilterInfo const &other)
{
  // merging two GenFilterInfos means that the numerator and
  // denominator from the original product need to besummed with
  // those in the product we are going to merge
  numEventsTried_ += other.numEventsTried(); 
  numEventsPassed_ += other.numEventsPassed();   

  sumPassPositiveWeights_ += other.sumPassPositiveWeights();
  sumPassPositiveWeights2_+= other.sumPassPositiveWeights2();
  sumPassNegativeWeights_ += other.sumPassNegativeWeights();
  sumPassNegativeWeights2_+= other.sumPassNegativeWeights2();

  sumFailPositiveWeights_ += other.sumFailPositiveWeights();
  sumFailPositiveWeights2_+= other.sumFailPositiveWeights2();
  sumFailNegativeWeights_ += other.sumFailNegativeWeights();
  sumFailNegativeWeights2_+= other.sumFailNegativeWeights2();
  return true;
}


double GenFilterInfo::filterWeightedEfficiency() const {

  double deno = sumWeights();
  double numr = sumPassWeights();

  return ( deno>1e-6? numr/deno: -1.);

}


double GenFilterInfo::filterWeightedEfficiencyError() const {

  double sum_plus = sumPositiveWeights();
  double sigma_plus = sum_plus > 1e-6?
    sqrt( 
  sumPassPositiveWeights2_ * sumFailPositiveWeights_* sumFailPositiveWeights_ + 
  sumFailPositiveWeights2_ * sumPassPositiveWeights_* sumPassPositiveWeights_)
    / sum_plus/sum_plus : 0;

  double sum_minus = sumNegativeWeights();
  double sigma_minus = sum_minus > 1e-6?
    sqrt( 
  sumPassNegativeWeights2_ * sumFailNegativeWeights_* sumFailNegativeWeights_ + 
  sumFailNegativeWeights2_ * sumPassNegativeWeights_* sumPassNegativeWeights_)
    / sum_minus/sum_minus : 0;
			    

  double denominator = sumWeights()*sumWeights();
  double numerator = sum_plus*sum_plus*sigma_plus*sigma_plus + 
    sum_minus*sum_minus*sigma_minus*sigma_minus;

  double sigma = denominator > 1e-6? sqrt(numerator/denominator):-1;

  return sigma;

}

double GenFilterInfo::filterWeightedEfficiencyErrorBinomial() const {

  return (filterEfficiency()>1e-6? 
	  (filterEfficiencyError()/filterEfficiency())*filterWeightedEfficiency():-1);
}
