#include "SimG4Core/Notification/interface/GenericComponent.h"
#include "Reflex/Reflex.h"
namespace frappe {

  bool Configurator::addComponent(const std::type_info & ti, void * v) {
    seal::reflex::Type t =  seal::reflex::Type::byTypeInfo(ti);
    std::string lazyname("frappe::LazyComponent<");
    lazyname+=t.name(seal::reflex::SCOPED);
    lazyname+=">";
    seal::reflex::Type lct =  seal::reflex::Type::byName(lazyname);
    seal::reflex::Type signature = 
	seal::reflex::Type::byName("void (seal::Context&,"+t.name(seal::reflex::SCOPED)+"*)"); 
    std::vector<void*> values(2); 
    values[0] = (void*)(&context());
    values[1] = (void*)(&v);
    return lct.construct(signature,values).address(); // it will register by itself...
  }
}
