#include "Utilities/StorageFactory/test/Test.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <errno.h>
#include <iostream>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

int main (int, char **) try {
  initTest();
  char pattern[] = "mkstemp-test-XXXXXX\0";
  struct stat status;
  mode_t previous_umask = umask(000);
  int fd = mkstemp(pattern);
  umask(previous_umask);
  if (fd == -1) {
    throw cms::Exception("TemporaryFile")
      << "Cannot create temporary file '" << pattern << "': "
      << strerror(errno) << " (error " << errno << ")";
  }
  int ret = fstat(fd, &status);
  unlink(pattern);
  if(ret != 0) {
    throw cms::Exception("TemporaryFile")
      << "Cannot fstat temporary file '" << pattern << "': "
      << strerror(errno) << " (error " << errno << ")";
  }
  mode_t mode = status.st_mode & 0777;
  if(mode != 0600) {
    throw cms::Exception("TemporaryFile")
      << "Temporary file '" << pattern << "': "
      << "created with mode " << std::oct << mode << " rather than 0600";
  }
  return EXIT_SUCCESS;
} catch(cms::Exception const& e) {
  std::cerr << e.explainSelf() << std::endl;
  return EXIT_FAILURE;
} catch(std::exception const& e) {
  std::cerr << e.what() << std::endl;
  return EXIT_FAILURE;
}
