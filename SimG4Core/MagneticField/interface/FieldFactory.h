#ifndef SimG4Core_FieldFactory_H
#define SimG4Core_FieldFactory_H

#include "SealKernel/Component.h"
#include "PluginManager/PluginFactory.h"

class FieldBuilder;
class LocalFieldManager;
class G4FieldManager;

class FieldFactory : public seal::PluginFactory<
    Field * (seal::Context *,const edm::ParameterSet & p) >
{
public:
    virtual ~FieldFactory();
    static FieldFactory * get(); 
    void build();
private:
    static FieldFactory s_instance;
    FieldFactory();
    FieldBuilder * theFieldBuilder;
    LocalFieldManager * theLocalFM;
};

#endif
