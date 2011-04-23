#include "Utilities/StorageFactory/interface/StorageMaker.h"
#include "Utilities/StorageFactory/interface/Storage.h"
#include "Utilities/StorageFactory/interface/IOFlags.h"
#include <cstdlib>

StorageMaker::StorageMaker (void)
{}

StorageMaker::~StorageMaker (void)
{}

void
StorageMaker::stagein (const std::string &/*proto*/,
                       const std::string &/*path*/)
{}

void
StorageMaker::setTimeout (unsigned int /*timeout*/)
{}

bool
StorageMaker::check (const std::string &proto,
		     const std::string &path,
		     IOOffset *size /* = 0 */)
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
  if (Storage *s = open (proto, path, mode))
  {
    if (size)
      *size = s->size ();

    s->close ();
    delete s;

    found = true;
  }

  return found;
}
