#include "Utilities/DavixAdaptor/interface/DavixFile.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <cassert>
#include <davix.hpp>
#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <unistd.h>
#include <vector>

using namespace Davix;

static Context *davix_context_s = NULL;

DavixFile::DavixFile(void) {}

DavixFile::DavixFile(const char *name, int flags /* = IOFlags::OpenRead */, int perms /* = 066 */) {
  open(name, flags, perms);
}

DavixFile::DavixFile(const std::string &name, int flags /* = IOFlags::OpenRead */,
                     int perms /* = 066 */) {
  open(name.c_str(), flags, perms);
}

DavixFile::~DavixFile(void) {
  close();
  return;
}

Context *DavixFile::getDavixInstance() {
  if (davix_context_s == NULL) {
    davix_context_s = new Context();
  }
  return davix_context_s;
}

void DavixFile::close(void) {
  if (davixPosix != NULL && m_fd != NULL) {
    DavixError *err;
    davixPosix->close(m_fd, &err);
    delete davixPosix;
  }
  return;
}

void DavixFile::abort(void) {
  if (davixPosix != NULL && m_fd != NULL) {
    DavixError *err;
    davixPosix->close(m_fd, &err);
    delete davixPosix;
  }
  return;
}

void DavixFile::create(const char *name, bool exclusive /* = false */, int perms /* = 066 */) {
  open(name, (IOFlags::OpenCreate | IOFlags::OpenWrite | IOFlags::OpenTruncate |
              (exclusive ? IOFlags::OpenExclusive : 0)),
       perms);
}

void DavixFile::create(const std::string &name, bool exclusive /* = false */,
                       int perms /* = 066 */) {
  open(name.c_str(), (IOFlags::OpenCreate | IOFlags::OpenWrite | IOFlags::OpenTruncate |
                      (exclusive ? IOFlags::OpenExclusive : 0)),
       perms);
}

void DavixFile::open(const std::string &name, int flags /* = IOFlags::OpenRead */,
                     int perms /* = 066 */) {
  open(name.c_str(), flags, perms);
}

void DavixFile::open(const char *name, int flags /* = IOFlags::OpenRead */, int perms /* = 066 */) {
  // Actual open
  if ((name == 0) || (*name == 0)) {
    edm::Exception ex(edm::errors::FileOpenError);
    ex << "Cannot open a file without name";
    ex.addContext("Calling DavixFile::open()");
    throw ex;
  }

  if ((flags & IOFlags::OpenRead) == 0) {
    edm::Exception ex(edm::errors::FileOpenError);
    ex << "Must open file '" << name << "' at least for read";
    ex.addContext("Calling DavixFile::open()");
    throw ex;
  }

  // Translate our flags to system flags
  int openflags = 0;

  if (flags & IOFlags::OpenRead)
    openflags |= O_RDONLY;

  DavixError *davixErr;
  davixPosix = new DavPosix(getDavixInstance());
  m_fd = davixPosix->open(NULL, name, openflags, &davixErr);
  m_name = name;
}

IOSize DavixFile::readv(IOBuffer *into, IOSize buffers) {
  assert(!buffers || into);

  // Davix does not support 0 buffers;
  if (buffers == 0)
    return 0;

  DavixError *davixErr;

  DavIOVecInput input_vector[buffers];
  DavIOVecOuput output_vector[buffers];
  IOSize total = 0; // Total requested bytes
  for (IOSize i = 0; i < buffers; ++i) {
    input_vector[i].diov_size = into[i].size();
    input_vector[i].diov_buffer = (char *)into[i].data();
    total += into[i].size();
  }

  ssize_t s = davixPosix->preadVec(m_fd, input_vector, output_vector, buffers, &davixErr);
  return total;
}

IOSize DavixFile::readv(IOPosBuffer *into, IOSize buffers) {
  assert(!buffers || into);

  // Davix does not support 0 buffers;
  if (buffers == 0)
    return 0;

  DavixError *davixErr;

  DavIOVecInput input_vector[buffers];
  DavIOVecOuput output_vector[buffers];
  IOSize total = 0;
  for (IOSize i = 0; i < buffers; ++i) {
    input_vector[i].diov_offset = into[i].offset();
    input_vector[i].diov_size = into[i].size();
    input_vector[i].diov_buffer = (char *)into[i].data();
    total += into[i].size();
  }
  ssize_t s = davixPosix->preadVec(m_fd, input_vector, output_vector, buffers, &davixErr);
  return total;
}

IOSize DavixFile::read(void *into, IOSize n) {
  DavixError *davixErr;
  davixPosix->fadvise(m_fd, 0, n, AdviseRandom);
  IOSize done = 0;
  while (done < n) {
    ssize_t s = davixPosix->read(m_fd, (char *)into + done, n - done, &davixErr);
    done += s;
  }
  return done;
}

IOSize DavixFile::write(const void *from, IOSize n) {
  cms::Exception ex("FileResizeError");
  ex << "DavixFile::resize(name='" << m_name << "') not implemented";
  throw ex;
}

IOOffset DavixFile::position(IOOffset offset, Relative whence /* = SET */) {
  DavixError *davixErr;
  IOOffset result;
  size_t mywhence = (whence == SET ? SEEK_SET : whence == CURRENT ? SEEK_CUR : SEEK_END);
  result = davixPosix->lseek(m_fd, offset, mywhence, &davixErr);
  return result;
}

void DavixFile::resize(IOOffset /* size */) {
  cms::Exception ex("FileResizeError");
  ex << "DavixFile::resize(name='" << m_name << "') not implemented";
  throw ex;
}
