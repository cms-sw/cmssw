#include <iostream>
#include <algorithm> 

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "SimDataFormats/GeneratorProducts/interface/GenRunInfoProduct.h"

using namespace edm;
using namespace std;

GenRunInfoProduct::GenRunInfoProduct() :
	externalFilterEfficiency_(-1.)
{
}

GenRunInfoProduct::GenRunInfoProduct(GenRunInfoProduct const &other) :
	internalXSec_(other.internalXSec_),
	externalXSecLO_(other.externalXSecLO_),
	externalXSecNLO_(other.externalXSecNLO_),
	externalFilterEfficiency_(other.externalFilterEfficiency_)
{
}

GenRunInfoProduct::~GenRunInfoProduct()
{
}

bool GenRunInfoProduct::mergeProduct(GenRunInfoProduct const &other)
{
	// if external xsec and filter efficiency are identical we allow merging
	// unfortunately we cannot merge on-the-fly xsec, as we do not have
	// information about the corresponding statistics (i.e. uncertainty)
	// so we just ignore it.
	// So in the end this merger is a dummy (just a safety check)

	if (externalXSecLO_ == other.externalXSecLO_ &&
	    externalXSecNLO_ == other.externalXSecNLO_ &&
	    externalFilterEfficiency_ == other.externalFilterEfficiency_)
		return true;

	edm::LogWarning("GenRunInfoProduct|ProductsNotMergeable")
		<< "You are merging runs with different cross-sections and/or "
		   "filter efficiencies (from GenRunInfoProduct)\n"
		   "The resulting cross-section will not be consistent." << std::endl;

	return true;
}

bool GenRunInfoProduct::isProductEqual(GenRunInfoProduct const &other) const
{
	return externalXSecLO_ == other.externalXSecLO_ &&
	       externalXSecNLO_ == other.externalXSecNLO_ &&
	       externalFilterEfficiency_ == other.externalFilterEfficiency_;
}
