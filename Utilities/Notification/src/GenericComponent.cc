#include "Utilities/Notification/interface/GenericComponent.h"
#include "Reflex/Type.h"
#include "Reflex/Object.h"

namespace frappe {

  bool Configurator::addComponent(const std::type_info & ti, void * v) {
    ROOT::Reflex::Type t =  ROOT::Reflex::Type::ByTypeInfo(ti);
    std::string lazyname("frappe::LazyComponent<");
    lazyname+=t.Name(ROOT::Reflex::SCOPED);
    lazyname+=">";
    ROOT::Reflex::Type lct =  ROOT::Reflex::Type::ByName(lazyname);
    ROOT::Reflex::Type signature = 
	ROOT::Reflex::Type::ByName("void (seal::Context&,"+t.Name(ROOT::Reflex::SCOPED)+"*)"); 
    std::vector<void*> values(2); 
    values[0] = (void*)(&context());
    values[1] = (void*)(&v);
    return lct.Construct(signature,values).Address(); // it will register by itself...
  }
}
