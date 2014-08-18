#include <iostream>
#include <algorithm> 

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "SimDataFormats/GeneratorProducts/interface/GenFilterInfo.h"

using namespace edm;
using namespace std;

GenFilterInfo::GenFilterInfo() :
  numPassPositiveEvents_(0),
  numPassNegativeEvents_(0),
  numFailPositiveEvents_(0),
  numFailNegativeEvents_(0),
  sumPassWeights_(0.),
  sumPassWeights2_(0.),
  sumFailWeights_(0.),
  sumFailWeights2_(0.)
{
}

GenFilterInfo::GenFilterInfo(unsigned int tried, unsigned int pass) :
  numPassPositiveEvents_(pass),
  numPassNegativeEvents_(0),
  numFailPositiveEvents_(tried-pass),
  numFailNegativeEvents_(0),
  sumPassWeights_(pass),
  sumPassWeights2_(pass),
  sumFailWeights_(tried-pass),
  sumFailWeights2_(tried-pass)
{
}

GenFilterInfo::GenFilterInfo(unsigned int passp, unsigned int passn, unsigned int failp, unsigned int failn,
			     double passw, double passw2, double failw, double failw2) :
  numPassPositiveEvents_(passp),
  numPassNegativeEvents_(passn),
  numFailPositiveEvents_(failp),
  numFailNegativeEvents_(failn),
  sumPassWeights_(passw),
  sumPassWeights2_(passw2),
  sumFailWeights_(failw),
  sumFailWeights2_(failw2)
{
}

GenFilterInfo::GenFilterInfo(const GenFilterInfo& other):
  numPassPositiveEvents_(other.numPassPositiveEvents_),
  numPassNegativeEvents_(other.numPassNegativeEvents_),
  numFailPositiveEvents_(other.numFailPositiveEvents_),
  numFailNegativeEvents_(other.numFailNegativeEvents_),
  sumPassWeights_(other.sumPassWeights_),
  sumPassWeights2_(other.sumPassWeights2_),
  sumFailWeights_(other.sumFailWeights_),
  sumFailWeights2_(other.sumFailWeights2_)
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

  numPassPositiveEvents_ += other.numPassPositiveEvents_;
  numPassNegativeEvents_ += other.numPassNegativeEvents_;
  numFailPositiveEvents_ += other.numFailPositiveEvents_;
  numFailNegativeEvents_ += other.numFailNegativeEvents_;
  sumPassWeights_        += other.sumPassWeights_;
  sumPassWeights2_       += other.sumPassWeights2_;
  sumFailWeights_        += other.sumFailWeights_;
  sumFailWeights2_       += other.sumFailWeights2_;

  return true;
}

double GenFilterInfo::filterEfficiency(int idwtup) const {
  double eff = -1;
  switch(idwtup) {
  case 3: case -3:
    eff = numEventsTotal() > 0 ? (double) numEventsPassed() / (double) numEventsTotal(): -1;
    break;
  default:
    eff = sumWeights() > 1e-6? sumPassWeights()/sumWeights(): -1;
    break;
  }
  return eff;

}

double GenFilterInfo::filterEfficiencyError(int idwtup) const {

  double efferr = -1;
  switch(idwtup) {
  case 3: case -3:
    {
      double effp  = numPositiveEvents() > 1e-6? 
	(double)numPassPositiveEvents()/(double)numPositiveEvents():0;
      double effp_err2 = numPositiveEvents() > 1e-6?
	(1-effp)*effp/(double)numPositiveEvents(): 0;

      double effn  = numNegativeEvents() > 1e-6? 
	(double)numPassNegativeEvents()/(double)numNegativeEvents():0;
      double effn_err2 = numNegativeEvents() > 1e-6? 
	(1-effn)*effn/(double)numNegativeEvents(): 0;

      efferr = numEventsTotal() > 0 ? sqrt ( 
					    ((double)numPositiveEvents()*(double)numPositiveEvents()*effp_err2 + 
					     (double)numNegativeEvents()*(double)numNegativeEvents()*effn_err2)
					    /(double)numEventsTotal()/(double)numEventsTotal()
					     ) : -1;
      break;
    }
  default:
    {
      double denominator = sumWeights()*sumWeights()*sumWeights()*sumWeights();
      double numerator =
	sumPassWeights2() * sumFailWeights()* sumFailWeights() +
	sumFailWeights2() * sumPassWeights()* sumPassWeights();
      efferr = denominator>1e-6? 
	sqrt(numerator/denominator):-1;
      break;
    }
  }

  return efferr;
}
