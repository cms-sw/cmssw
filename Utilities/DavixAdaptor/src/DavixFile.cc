#include "Utilities/DavixAdaptor/interface/DavixFile.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <cassert>
#include <davix.hpp>
#include <cerrno>
#include <fcntl.h>
#include <cstdlib>
#include <unistd.h>
#include <vector>
#include <mutex>

static std::once_flag davixDebugInit;

using namespace Davix;

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

void DavixFile::close(void) {
  if (m_davixPosix && m_fd) {
    auto davixPosix = std::move(m_davixPosix);
    DavixError *err = nullptr;
    davixPosix->close(m_fd, &err);
    m_fd = nullptr;
    if (err) {
      std::unique_ptr<DavixError> davixErrManaged(err);
      cms::Exception ex("FileCloseError");
      ex << "Davix::close(name='" << m_name << ") failed with error "
         << err->getErrMsg().c_str() << " and error code " << err->getStatus();
      ex.addContext("Calling DavixFile::close()");
      throw ex;
    }
  }
  return;
}

void DavixFile::abort(void) {
  if (m_davixPosix && m_fd) {
    DavixError *err = nullptr;
    m_davixPosix->close(m_fd, &err);
    if (err) {
      std::unique_ptr<DavixError> davixErrManaged(err);
      cms::Exception ex("FileAbortError");
      ex << "Davix::abort(name='" << m_name << ") failed with error "
         << err->getErrMsg().c_str() << " and error code " << err->getStatus();
      ex.addContext("Calling DavixFile::abort()");
      throw ex;
    }
  }
  return;
}

void DavixFile::configureDavixLogLevel() {
  long logLevel = 0;
  char *logptr = nullptr;
  char const * const davixDebug = getenv("Davix_Debug");
  if (davixDebug != nullptr) {
    logLevel = strtol(davixDebug, &logptr, 0);
    if (errno) {
      edm::LogWarning("DavixFile") << "Got error while converting "
                                   << "Davix_Debug env variable to integer. "
                                      "Will use default log level 0";
      logLevel = 0;
    }
    if (logptr == davixDebug) {
      edm::LogWarning("DavixFile") << "Failed to convert to integer "
                                   << "Davix_Debug env variable; Will use default log level 0";
      logLevel = 0;
    } else if (*logptr != '\0') {
      edm::LogWarning("DavixFile") << "Failed to parse extra junk "
                                   << "from Davix_Debug env variable. Will use default log level 0";
      logLevel = 0;
    }
  }
  switch (logLevel) {
  case 0:
    std::call_once(davixDebugInit, davix_set_log_level, 0);
    break;
  case 1:
    std::call_once(davixDebugInit, davix_set_log_level, DAVIX_LOG_WARNING);
    break;
  case 2:
    std::call_once(davixDebugInit, davix_set_log_level, DAVIX_LOG_VERBOSE);
    break;
  case 3:
    std::call_once(davixDebugInit, davix_set_log_level, DAVIX_LOG_DEBUG);
    break;
  default:
    std::call_once(davixDebugInit, davix_set_log_level, DAVIX_LOG_ALL);
    break;
  }
}

static int X509Authentication(void *userdata, const SessionInfo &info, X509Credential *cert,
                              DavixError **davixErr) {
  std::string ucert, ukey;
  char default_proxy[64];
  snprintf(default_proxy, sizeof(default_proxy), "/tmp/x509up_u%d", geteuid());
  // X509_USER_PROXY
  if (getenv("X509_USER_PROXY")) {
    edm::LogInfo("DavixFile") << "X509_USER_PROXY found in environment."
                              << " Will use it for authentication";
    ucert = ukey = getenv("X509_USER_PROXY");
  }
  // Default proxy location
  else if (access(default_proxy, R_OK) == 0) {
    edm::LogInfo("DavixFile") << "Found proxy in default location " << default_proxy
                              << " Will use it for authentication";
    ucert = ukey = default_proxy;
  }
  // X509_USER_CERT
  else if (getenv("X509_USER_CERT")) {
    ucert = getenv("X509_USER_CERT");
  }
  // X509_USER_KEY only if X509_USER_CERT was found
  if (!ucert.empty() && getenv("X509_USER_KEY")) {
    edm::LogInfo("DavixFile") << "X509_USER_{CERT|KEY} found in environment"
                              << " Will use it for authentication";
    ukey = getenv("X509_USER_KEY");
  }
  // Check if vars are set...
  if (ucert.empty() || ukey.empty()) {
    edm::LogWarning("DavixFile") << "Was not able to find proxy in $X509_USER_PROXY, "
                                 << "X509_USER_{CERT|KEY} or default proxy creation location. "
                                 << "Will try without authentication";
    return -1;
  }
  return cert->loadFromFilePEM(ukey, ucert, "", davixErr);
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
  if ((name == nullptr) || (*name == 0)) {
    edm::Exception ex(edm::errors::FileOpenError);
    ex << "Cannot open a file without name";
    ex.addContext("Calling DavixFile::open()");
    throw ex;
  }
  m_name = name;

  if ((flags & IOFlags::OpenRead) == 0) {
    edm::Exception ex(edm::errors::FileOpenError);
    ex << "Must open file '" << name << "' at least for read";
    ex.addContext("Calling DavixFile::open()");
    throw ex;
  }

  if (m_davixPosix && m_fd) {
    edm::Exception ex(edm::errors::FileOpenError);
    ex << "Davix::open(name='" << m_name << "') failed on already open file";
    ex.addContext("Calling DavixFile::open()");
    throw ex;
  }
  configureDavixLogLevel();
  // Translate our flags to system flags
  int openflags = 0;

  if (flags & IOFlags::OpenRead)
    openflags |= O_RDONLY;

  DavixError *davixErr = nullptr;
  RequestParams davixReqParams;
  // Set up X509 authentication
  davixReqParams.setClientCertCallbackX509(&X509Authentication, nullptr);
  // Set also CERT_DIR if it is set in envinroment, otherwise use default
  const char *cert_dir = nullptr;
  if ((cert_dir = getenv("X509_CERT_DIR")) == nullptr)
    cert_dir = "/etc/grid-security/certificates";
  davixReqParams.addCertificateAuthorityPath(cert_dir);

  m_davixPosix = std::make_unique<DavPosix>(new Context());
  m_fd = m_davixPosix->open(&davixReqParams, name, openflags, &davixErr);

  // Check Davix Error
  if (davixErr) {
    std::unique_ptr<DavixError> davixErrManaged(davixErr);
    edm::Exception ex(edm::errors::FileOpenError);
    ex << "Davix::open(name='" << m_name << "') failed with "
       << "error '" << davixErr->getErrMsg().c_str() << " and error code " << davixErr->getStatus();
    ex.addContext("Calling DavixFile::open()");
    throw ex;
  }
  if (!m_fd) {
    edm::Exception ex(edm::errors::FileOpenError);
    ex << "Davix::open(name='" << m_name << "') failed as fd is NULL";
    ex.addContext("Calling DavixFile::open()");
    throw ex;
  }
}

IOSize DavixFile::readv(IOBuffer *into, IOSize buffers) {
  assert(!buffers || into);

  // Davix does not support 0 buffers;
  if (buffers == 0)
    return 0;

  DavixError *davixErr = nullptr;

  std::vector<DavIOVecInput> input_vector(buffers);
  std::vector<DavIOVecOuput> output_vector(buffers);
  IOSize total = 0; // Total requested bytes
  for (IOSize i = 0; i < buffers; ++i) {
    input_vector[i].diov_size = into[i].size();
    input_vector[i].diov_buffer = static_cast<char *>(into[i].data());
    total += into[i].size();
  }

  ssize_t s = m_davixPosix->preadVec(m_fd, input_vector.data(), output_vector.data(), buffers, &davixErr);
  if (davixErr) {
    std::unique_ptr<DavixError> davixErrManaged(davixErr);
    edm::Exception ex(edm::errors::FileReadError);
    ex << "Davix::readv(name='" << m_name << "', buffers=" << (buffers)
       << ") failed with error " << davixErr->getErrMsg().c_str() << " and error code "
       << davixErr->getStatus() << " and call returned " << s << " bytes";
    ex.addContext("Calling DavixFile::readv()");
    throw ex;
  }
  // Davix limits number of requests sent to the server
  // to improve performance and it does range coalescing.
  // So we can`t check what preadVec returns with what was requested.
  // Example: If two ranges are overlapping, [10, 20] and [20, 30] which is
  // coalesced into [10, 30] and it will contain one less byte than was requested.
  // Only check if returned val <= 0 and make proper actions.
  if (s < 0) {
    edm::Exception ex(edm::errors::FileReadError);
    ex << "Davix::readv(name='" << m_name << "') failed and call returned " << s;
    ex.addContext("Calling DavixFile::readv()");
    throw ex;
  } else if (s == 0) {
    // end of file
    return 0;
  }
  return total;
}

IOSize DavixFile::readv(IOPosBuffer *into, IOSize buffers) {
  assert(!buffers || into);

  // Davix does not support 0 buffers;
  if (buffers == 0)
    return 0;

  DavixError *davixErr = nullptr;

  std::vector<DavIOVecInput> input_vector(buffers);
  std::vector<DavIOVecOuput> output_vector(buffers);
  IOSize total = 0;
  for (IOSize i = 0; i < buffers; ++i) {
    input_vector[i].diov_offset = into[i].offset();
    input_vector[i].diov_size = into[i].size();
    input_vector[i].diov_buffer = static_cast<char *>(into[i].data());
    total += into[i].size();
  }
  ssize_t s = m_davixPosix->preadVec(m_fd, input_vector.data(), output_vector.data(), buffers, &davixErr);
  if (davixErr) {
    std::unique_ptr<DavixError> davixErrManaged(davixErr);
    edm::Exception ex(edm::errors::FileReadError);
    ex << "Davix::readv(name='" << m_name << "', n=" << buffers << ") failed with error "
       << davixErr->getErrMsg().c_str() << " and error code " << davixErr->getStatus()
       << " and call returned " << s << " bytes";
    ex.addContext("Calling DavixFile::readv()");
    throw ex;
  }
  // Davix limits number of requests sent to the server
  // to improve performance and it does range coalescing.
  // So we can`t check what preadVec returns with what was requested.
  // Example: If two ranges are overlapping, [10, 20] and [20, 30] which is
  // coalesced into [10, 30] and it will contain one less byte than was requested.
  // Only check if returned val <= 0 and make proper actions.
  if (s < 0) {
    edm::Exception ex(edm::errors::FileReadError);
    ex << "Davix::readv(name='" << m_name << "', n=" << buffers
       << ") failed and call returned " << s;
    ex.addContext("Calling DavixFile::readv()");
    throw ex;
  } else if (s == 0) {
    // end of file
    return 0;
  }
  return total;
}

IOSize DavixFile::read(void *into, IOSize n) {
  DavixError *davixErr = nullptr;
  m_davixPosix->fadvise(m_fd, 0, n, AdviseRandom);
  IOSize done = 0;
  while (done < n) {
    ssize_t s = m_davixPosix->read(m_fd, (char *)into + done, n - done, &davixErr);
    if (davixErr) {
      std::unique_ptr<DavixError> davixErrManaged(davixErr);
      edm::Exception ex(edm::errors::FileReadError);
      ex << "Davix::read(name='" << m_name << "', n=" << (n - done)
         << ") failed with error " << davixErr->getErrMsg().c_str() << " and error code "
         << davixErr->getStatus() << " and call returned " << s << " bytes";
      ex.addContext("Calling DavixFile::read()");
      throw ex;
    }
    if (s < 0) {
      edm::Exception ex(edm::errors::FileReadError);
      ex << "Davix::read(name='" << m_name << "', n=" << (n - done)
         << ") failed and call returned " << s;
      ex.addContext("Calling DavixFile::read()");
      throw ex;
    } else if (s == 0) {
      // end of file
      break;
    }
    done += s;
  }
  return done;
}

IOSize DavixFile::write(const void *from, IOSize n) {
  edm::Exception ex(edm::errors::FileWriteError);
  ex << "DavixFile::write(name='" << m_name << "') not implemented";
  ex.addContext("Calling DavixFile::write()");
  throw ex;
}

IOOffset DavixFile::position(IOOffset offset, Relative whence /* = SET */) {
  DavixError *davixErr = nullptr;
  if (whence != CURRENT && whence != SET && whence != END) {
    cms::Exception ex("FilePositionError");
    ex << "DavixFile::position() called with incorrect 'whence' parameter";
    ex.addContext("Calling DavixFile::position()");
    throw ex;
  }
  IOOffset result;
  size_t mywhence = (whence == SET ? SEEK_SET : whence == CURRENT ? SEEK_CUR : SEEK_END);

  if ((result = m_davixPosix->lseek(m_fd, offset, mywhence, &davixErr)) == -1) {
    cms::Exception ex("FilePositionError");
    ex << "Davix::lseek(name='" << m_name << "', offset=" << offset
       << ", whence=" << mywhence << ") failed with error " << davixErr->getErrMsg().c_str()
       << " and error code " << davixErr->getStatus() << " and "
       << "call returned " << result;
    ex.addContext("Calling DavixFile::position()");
    throw ex;
  }

  return result;
}

void DavixFile::resize(IOOffset /* size */) {
  cms::Exception ex("FileResizeError");
  ex << "DavixFile::resize(name='" << m_name << "') not implemented";
  ex.addContext("Calling DavixFile::resize()");
  throw ex;
}
