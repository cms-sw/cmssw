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
#include "SimDataFormats/GeneratorProducts/interface/HepMC3Product.h"

#include "SimDataFormats/GeneratorProducts/interface/GenRunInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenFilterInfo.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct3.h"
#include "SimDataFormats/GeneratorProducts/interface/GenLumiInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenLumiInfoHeader.h"
#include "SimDataFormats/GeneratorProducts/interface/ExternalGeneratorLumiInfo.h"
#include "SimDataFormats/GeneratorProducts/interface/ExternalGeneratorEventInfo.h"

#include <HepMC/GenRanges.h>

//needed for backward compatibility between HepMC 2.06.xx and 2.05.yy
namespace hepmc_rootio {
  void add_to_particles_in(HepMC::GenVertex*, HepMC::GenParticle*);
  void clear_particles_in(HepMC::GenVertex*);

  inline void weightcontainer_set_default_names(unsigned int n,
                                                std::map<std::string, HepMC::WeightContainer::size_type>& names) {
    std::ostringstream name;
    for (HepMC::WeightContainer::size_type count = 0; count < n; ++count) {
      name.str(std::string());
      name << count;
      names[name.str()] = count;
    }
  }
}  // namespace hepmc_rootio

// needed for backwards compatibility for auto_ptr<gen::PdfInfo>
namespace pdfinfo_autoptr_rootio {
  // Following the pattern in DataFormats/TestObjects/src/classes.h
  template <typename T>
  struct deprecated_auto_ptr {
    // We use compat_auto_ptr only to assign the wrapped raw pointer
    // to a unique pointer in an I/O customization rule.
    // Therefore, we don't delete on destruction (because ownership
    // gets transferred to the unique pointer).

    // ~deprecated_auto_ptr() { delete _M_ptr; }

    T* _M_ptr = nullptr;
  };
}  // namespace pdfinfo_autoptr_rootio
