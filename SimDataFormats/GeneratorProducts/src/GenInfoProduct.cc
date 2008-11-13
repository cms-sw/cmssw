#include <iostream>
#include <algorithm> 

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "SimDataFormats/GeneratorProducts/interface/GenInfoProduct.h"

using namespace edm;
using namespace std;

GenInfoProduct::GenInfoProduct( double cross_section ) {
    cs_ = cross_section;
    cs2_ = 0;
    fe_ = 0;
}


// copy constructor
GenInfoProduct::GenInfoProduct(GenInfoProduct const& other) :
  cs_(other.cross_section()), cs2_(other.external_cross_section()), fe_(other.filter_efficiency()) {
}


bool GenInfoProduct::mergeProduct(GenInfoProduct const& other)
{
  // if external xsec and filter efficiency are identical we allow merging
  // unfortunately we cannot merge on-the-fly xsec, as we do not have
  // information about the corresponding statistics (i.e. uncertainty)
  // so we just ignore it.
  // So in the end this merger is a dummy (just a safety check)

  if (cs2_ == other.cs2_ && fe_ == other.fe_)
    return true;

  edm::LogWarning("GenInfoProduct|ProductsNotMergeable")
        << "You are merging runs with different cross-sections and/or "
           "filter efficiencies (from GenInfoProduct)\n"
           "The resulting cross-section will not be consistent."
        << std::endl;

  return true;
}
