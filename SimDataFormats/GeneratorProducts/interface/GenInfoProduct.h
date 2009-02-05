#ifndef SimDataFormats_GeneratorProducts_GenInfoProduct_h
#define SimDataFormats_GeneratorProducts_GenInfoProduct_h

#warning SimDataFormats/GeneratorProducts/GenInfoProduct.h is deprecated and will go away.  Please use the new GenRunInfoProduct class instead!

#include "SimDataFormats/GeneratorProducts/interface/GenRunInfoProduct.h"

namespace edm {
	typedef GenRunInfoProduct GenInfoProduct __attribute__ ((deprecated));
}

#endif // SimDataFormats_GeneratorProducts_GenInfoProduct_h
