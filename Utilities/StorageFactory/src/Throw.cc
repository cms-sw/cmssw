#include "Utilities/StorageFactory/src/Throw.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <ostream>
#include <cstring>

void
throwStorageError (const char *context, const char *call, int error)
{
  throw cms::Exception (context)
    << call << " failed with system error '"
    << strerror (error) << "' (error code " << error << ")";
}
