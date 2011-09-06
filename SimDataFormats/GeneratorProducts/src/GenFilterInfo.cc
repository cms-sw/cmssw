#include <iostream>
#include <algorithm> 

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "SimDataFormats/GeneratorProducts/interface/GenFilterInfo.h"

using namespace edm;
using namespace std;

GenFilterInfo::GenFilterInfo() :
  numEventsTried_(0),
  numEventsPassed_(0)
{
}

GenFilterInfo::GenFilterInfo(unsigned int tried_, unsigned int passed_) :
  numEventsTried_(tried_),
  numEventsPassed_(passed_)
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
  return true;
}

