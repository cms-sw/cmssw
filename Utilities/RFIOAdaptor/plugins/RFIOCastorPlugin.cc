#ifndef __APPLE__
#include "Utilities/RFIOAdaptor/interface/RFIOPluginFactory.h"
DEFINE_EDM_PLUGIN (RFIOPluginFactory,RFIODummyFile,"castor");
#else // __APPLE__
#include <unistd.h>
extern "C" {
int
closefunc(int s)
{
	return(close(s));
}
}
int dummyCastorPlugin(){return 0;}
#endif 
