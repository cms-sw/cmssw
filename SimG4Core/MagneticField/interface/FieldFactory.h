#ifndef SimG4Core_FieldFactory_H
#define SimG4Core_FieldFactory_H

#include "Utilities/Notification/interface/DispatcherObserver.h"
#include "SimG4Core/Geometry/interface/DDDWorld.h"

class FieldBuilder;
class LocalFieldManager;
class G4FieldManager;

class FieldFactory : private frappe::Observer<DDDWorld>
{
public:
    typedef frappe::Observer<DDDWorld> super;
    FieldFactory(seal::Context * ic, const std::string & iname);
    ~FieldFactory();
protected:
    virtual void update(const DDDWorld * w);    
private:
    FieldBuilder * theFieldBuilder;
    LocalFieldManager * theLocalFM;
};

#endif
