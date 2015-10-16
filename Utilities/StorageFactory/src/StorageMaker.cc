#include "Utilities/StorageFactory/interface/StorageMaker.h"
#include "Utilities/StorageFactory/interface/Storage.h"
#include "Utilities/StorageFactory/interface/IOFlags.h"
#include <cstdlib>

void
StorageMaker::stagein (const std::string &/*proto*/,
                       const std::string &/*path*/,
                       const AuxSettings& ) const
{}

bool
StorageMaker::check (const std::string &proto,
		     const std::string &path,
         const AuxSettings& aux,
		     IOOffset *size /* = 0 */) const
{
  // Fallback method is to open the file and check its
  // size.  Because grid jobs run in a directory where
  // there is usually more space than in /tmp, and that
  // directory is automatically cleaned up, open up the
  // temporary files in the current directory.  If the
  // file is downloaded, it will delete itself in the
  // destructor or close method.
  bool found = false;
  int mode = IOFlags::OpenRead | IOFlags::OpenUnbuffered;
  if (auto s = open (proto, path, mode, aux))
  {
    if (size)
      *size = s->size ();

    s->close ();

    found = true;
  }

  return found;
}
