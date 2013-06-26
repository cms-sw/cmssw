#include <utility>
#include <vector>
#include <map>
#include <set>

#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/RefVector.h"

#include "SimDataFormats/GeneratorProducts/interface/LHEEventProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/LHERunInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/LHEXMLStringProduct.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "SimDataFormats/GeneratorProducts/interface/GenRunInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenFilterInfo.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"

#include <HepMC/GenRanges.h>

namespace {
	struct dictionary {
		// HepMC externals used in HepMCProduct

		HepMC::GenVertex dummy11;
		HepMC::GenParticle dummy22;
        HepMC::GenCrossSection dummy33;
		//Some member functions will not compile when using HepMCConfig as the parameter
		//HepPDT::DecayDataT<HepMCConfig> dd1;
		// lack of a public destructor plus other problems keeps us from generating this dictionary
		//HepPDT::ParticleDataT<HepMCConfig> pd1;
		std::map<int, HepMC::GenVertex*> m_vertex_barcodes;
		std::map<int, HepMC::GenParticle*> m_particle_barcodes;
		std::pair<const int, HepMC::GenVertex*> prgv1;
		std::pair<const int, HepMC::GenParticle*> prgp1;
		std::map<int, HepMC::GenVertex*, std::greater<int> > dummy777;
		std::vector<HepMC::GenParticle*> m_particles_out;
		std::vector<HepMC::GenParticle*> m_particles_in;

		// HepMCProduct

		edm::Wrapper<edm::HepMCProduct> m_wrapper;
                std::vector<const edm::HepMCProduct*> m_vector_const_ptr;
		edm::Ref<edm::HepMCProduct, HepMC::GenParticle> refGen;
		edm::Ref<edm::HepMCProduct, HepMC::GenVertex> refVert;

		edm::RefVector<edm::HepMCProduct, HepMC::GenParticle> refVGen;
		edm::RefVector<edm::HepMCProduct, HepMC::GenVertex> refVVert;

		//std::iterator<std::forward_iterator_tag, HepMC::GenVertex*, int, HepMC::GenVertex**, HepMC::GenVertex*&> itr1;
		//std::iterator<std::forward_iterator_tag, HepMC::GenParticle*, int, HepMC::GenParticle**, HepMC::GenParticle*&> itr2;

		// GenInfoProduct

		edm::Wrapper<GenRunInfoProduct> wgenruninfo;
		edm::Wrapper<GenFilterInfo> wgenfilterinfo;
		edm::Wrapper<GenEventInfoProduct> wgeneventinfo;

		// LHE products

		edm::Wrapper<LHERunInfoProduct>	wcommon;
		edm::Wrapper<LHEEventProduct>	wevent;
                edm::Wrapper<LHEXMLStringProduct> wstring;
	};
}

//needed for backward compatibility between HepMC 2.06.xx and 2.05.yy
namespace hepmc_rootio {
  inline void weightcontainer_set_default_names(unsigned int n, std::map<std::string,HepMC::WeightContainer::size_type>& names) {
      std::ostringstream name;
      for ( HepMC::WeightContainer::size_type count = 0; count<n; ++count ) 
      { 
      name.str(std::string());
      name << count;
      names[name.str()] = count;
      }
  }
}

