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

bool GenRunInfoProduct::isProductEqual(GenRunInfoProduct const &other) const
{
	bool result =  externalXSecLO_ == other.externalXSecLO_ &&
                       externalXSecNLO_ == other.externalXSecNLO_ &&
	               externalFilterEfficiency_ == other.externalFilterEfficiency_;
        if( not result) {
          edm::LogWarning("GenRunInfoProduct|ProductsNotMergeable")
            << "You are merging runs with different cross-sections and/or "
               "filter efficiencies (from GenRunInfoProduct)\n"
               "The resulting cross-section will not be consistent." << std::endl;
        }

        return result;
}
