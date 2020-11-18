#ifdef __CINT_
#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;
#pragma link C++ nestedclasses;
#pragma link C++ class validHit + ;
#pragma link C++ class vector < validHit> + ;
#endif /* __CINT__ */

#include "Validation/HGCalValidation/interface/validHit.h"
#include "DataFormats/Common/interface/AssociationVector.h"
#include "DataFormats/Common/interface/AssociationMap.h"

validHit vh;
std::vector<validHit> vvh;
edm::Wrapper<std::vector<validHit> > wvvh;
