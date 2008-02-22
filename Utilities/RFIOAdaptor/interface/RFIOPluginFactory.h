#ifndef RFIO_ADAPTOR_RFIO_PLUGIN_FACTORY_H
# define RFIO_ADAPTOR_RFIO_PLUGIN_FACTORY_H

//<<<<<< INCLUDES                                                       >>>>>>

#include "FWCore/PluginManager/interface/PluginFactory.h"

//<<<<<< PUBLIC DEFINES                                                 >>>>>>
//<<<<<< PUBLIC CONSTANTS                                               >>>>>>
//<<<<<< PUBLIC TYPES                                                   >>>>>>
//<<<<<< PUBLIC VARIABLES                                               >>>>>>
//<<<<<< PUBLIC FUNCTIONS                                               >>>>>>
//<<<<<< CLASS DECLARATIONS                                             >>>>>>
class RFIODummyFile 
{
public:
    RFIODummyFile();
};

typedef edmplugin::PluginFactory<RFIODummyFile*(void)> RFIOPluginFactory;

//<<<<<< INLINE PUBLIC FUNCTIONS                                        >>>>>>
//<<<<<< INLINE MEMBER FUNCTIONS                                        >>>>>>

#endif // RFIO_ADAPTOR_RFIO_PLUGIN_FACTORY_H
