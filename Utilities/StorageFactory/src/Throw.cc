#include "Utilities/StorageFactory/src/Throw.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <ostream>
#include <cstring>

void
throwStorageError (const char* category, 
                   const char *context,
                   const char *call, int error)
{  
  cms::Exception ex(category);
  ex << call << " failed with system error '"
     << strerror (error) << "' (error code " << error << ")";
  ex.addContext(context);
  throw ex;
}

void
throwStorageError (edm::errors::ErrorCodes category, 
                   const char *context,
                   const char *call, int error)
{  
  edm::Exception ex(category);
  ex << call << " failed with system error '"
     << strerror (error) << "' (error code " << error << ")";
  ex.addContext(context);
  throw ex;
}
