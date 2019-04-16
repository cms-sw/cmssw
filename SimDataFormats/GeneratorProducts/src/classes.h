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
#include "SimDataFormats/GeneratorProducts/interface/GenLumiInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenLumiInfoHeader.h"
#include <HepMC/GenRanges.h>

//needed for backward compatibility between HepMC 2.06.xx and 2.05.yy
namespace hepmc_rootio {
  void add_to_particles_in(HepMC::GenVertex*, HepMC::GenParticle*);
  void clear_particles_in(HepMC::GenVertex*);

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

