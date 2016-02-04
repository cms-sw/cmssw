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

