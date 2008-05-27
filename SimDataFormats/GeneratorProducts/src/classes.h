#include "DataFormats/Common/interface/Wrapper.h"

#include "SimDataFormats/GeneratorProducts/interface/LHEEventProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/LHERunInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/AlpgenInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/AlpWgtFileInfoProduct.h"

namespace {
	namespace {
		edm::Wrapper<LHERunInfoProduct>			wcommon;
		edm::Wrapper<LHEEventProduct>			wevent;
		edm::Wrapper<edm::AlpgenInfoProduct>		walpinfo;
		edm::Wrapper<edm::AlpWgtFileInfoProduct>	walpwgtfile;
	}
}
