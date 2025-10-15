
#include "TrivialSerialisation/Common/interface/TrivialSerialiserSourceFactory.h"
#include "TrivialSerialisation/Common/interface/TrivialSerialiserSource.h"
#include "TrivialSerialisation/Common/interface/TrivialSerialiser.h"
#include "DataFormats/PortableTestObjects/interface/TestHostObject.h"
#include "DataFormats/Portable/interface/PortableCollection.h"
#include "DataFormats/PortableTestObjects/interface/TestSoA.h"
#include "DataFormats/Provenance/interface/EventID.h"

DEFINE_EDM_PLUGIN(ngt::TrivialSerialiserSourceFactory, ngt::TrivialSerialiserSource<int>, typeid(int).name());
DEFINE_EDM_PLUGIN(ngt::TrivialSerialiserSourceFactory,
                  ngt::TrivialSerialiserSource<unsigned short>,
                  typeid(unsigned short).name());

using basic_string = std::basic_string<char, std::char_traits<char>>;
DEFINE_EDM_PLUGIN(ngt::TrivialSerialiserSourceFactory,
                  ngt::TrivialSerialiserSource<basic_string>,
                  typeid(basic_string).name());

DEFINE_EDM_PLUGIN(ngt::TrivialSerialiserSourceFactory,
                  ngt::TrivialSerialiserSource<PortableHostObject<portabletest::TestStruct>>,
                  typeid(PortableHostObject<portabletest::TestStruct>).name());
DEFINE_EDM_PLUGIN(ngt::TrivialSerialiserSourceFactory,
                  ngt::TrivialSerialiserSource<edm::EventID>,
                  typeid(edm::EventID).name());

using portablehostcollection = PortableHostCollection<portabletest::TestSoALayout<128, false>>;
DEFINE_EDM_PLUGIN(ngt::TrivialSerialiserSourceFactory,
                  ngt::TrivialSerialiserSource<portablehostcollection>,
                  typeid(portablehostcollection).name());

using portablehostmulticollection2 =
    PortableHostMultiCollection<portabletest::TestSoALayout<128, false>, portabletest::TestSoALayout2<128, false>>;
DEFINE_EDM_PLUGIN(ngt::TrivialSerialiserSourceFactory,
                  ngt::TrivialSerialiserSource<portablehostmulticollection2>,
                  typeid(portablehostmulticollection2).name());

using portablehostmulticollection3 = PortableHostMultiCollection<portabletest::TestSoALayout<128, false>,
                                                                 portabletest::TestSoALayout2<128, false>,
                                                                 portabletest::TestSoALayout3<128, false>>;
DEFINE_EDM_PLUGIN(ngt::TrivialSerialiserSourceFactory,
                  ngt::TrivialSerialiserSource<portablehostmulticollection3>,
                  typeid(portablehostmulticollection3).name());
