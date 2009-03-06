#include "DataFormats/Common/interface/Wrapper.h"

#include "SimDataFormats/GeneratorProducts/interface/LHEEventProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/LHERunInfoProduct.h"

namespace {
	namespace {
		edm::Wrapper<LHERunInfoProduct>	wcommon;
		edm::Wrapper<LHEEventProduct>	wevent;
	}
}
