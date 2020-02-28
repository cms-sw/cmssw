#include "Utilities/XrdAdaptor/src/XrdFile.h"
#include "Utilities/XrdAdaptor/src/XrdRequestManager.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Likely.h"
#include <vector>
#include <sstream>
#include <iostream>
#include <cassert>
#include <chrono>

using namespace XrdAdaptor;

// To be re-enabled when the monitoring interface is back.
//static const char *kCrabJobIdEnv = "CRAB_UNIQUE_JOB_ID";

#define XRD_CL_MAX_CHUNK 512 * 1024
#define XRD_CL_MAX_SIZE 1024

#define XRD_CL_MAX_READ_SIZE (8 * 1024 * 1024)

XrdFile::XrdFile(void) : m_offset(0), m_size(-1), m_close(false), m_name(), m_op_count(0) {}

XrdFile::XrdFile(const char *name, int flags /* = IOFlags::OpenRead */, int perms /* = 066 */)
    : m_offset(0), m_size(-1), m_close(false), m_name(), m_op_count(0) {
  open(name, flags, perms);
}

XrdFile::XrdFile(const std::string &name, int flags /* = IOFlags::OpenRead */, int perms /* = 066 */)
    : m_offset(0), m_size(-1), m_close(false), m_name(), m_op_count(0) {
  open(name.c_str(), flags, perms);
}

XrdFile::~XrdFile(void) {
  if (m_close)
    edm::LogError("XrdFileError") << "Destructor called on XROOTD file '" << m_name << "' but the file is still open";
}

//////////////////////////////////////////////////////////////////////
void XrdFile::create(const char *name, bool exclusive /* = false */, int perms /* = 066 */) {
  open(name,
       (IOFlags::OpenCreate | IOFlags::OpenWrite | IOFlags::OpenTruncate | (exclusive ? IOFlags::OpenExclusive : 0)),
       perms);
}

void XrdFile::create(const std::string &name, bool exclusive /* = false */, int perms /* = 066 */) {
  open(name.c_str(),
       (IOFlags::OpenCreate | IOFlags::OpenWrite | IOFlags::OpenTruncate | (exclusive ? IOFlags::OpenExclusive : 0)),
       perms);
}

void XrdFile::open(const std::string &name, int flags /* = IOFlags::OpenRead */, int perms /* = 066 */) {
  open(name.c_str(), flags, perms);
}

void XrdFile::open(const char *name, int flags /* = IOFlags::OpenRead */, int perms /* = 066 */) {
  // Actual open
  if ((name == nullptr) || (*name == 0)) {
    edm::Exception ex(edm::errors::FileOpenError);
    ex << "Cannot open a file without a name";
    ex.addContext("Calling XrdFile::open()");
    throw ex;
  }
  if ((flags & (IOFlags::OpenRead | IOFlags::OpenWrite)) == 0) {
    edm::Exception ex(edm::errors::FileOpenError);
    ex << "Must open file '" << name << "' at least for read or write";
    ex.addContext("Calling XrdFile::open()");
    throw ex;
  }

  // Translate our flags to system flags
  XrdCl::OpenFlags::Flags openflags = XrdCl::OpenFlags::None;

  if (flags & IOFlags::OpenWrite)
    openflags |= XrdCl::OpenFlags::Update;
  else if (flags & IOFlags::OpenRead)
    openflags |= XrdCl::OpenFlags::Read;

  if (flags & IOFlags::OpenAppend) {
    edm::Exception ex(edm::errors::FileOpenError);
    ex << "Opening file '" << name << "' in append mode not supported";
    ex.addContext("Calling XrdFile::open()");
    throw ex;
  }

  if (flags & IOFlags::OpenCreate) {
    if (!(flags & IOFlags::OpenExclusive))
      openflags |= XrdCl::OpenFlags::Delete;
    openflags |= XrdCl::OpenFlags::New;
    openflags |= XrdCl::OpenFlags::MakePath;
  }

  if ((flags & IOFlags::OpenTruncate) && (flags & IOFlags::OpenWrite))
    openflags |= XrdCl::OpenFlags::Delete;

  // Translate mode flags
  XrdCl::Access::Mode modeflags = XrdCl::Access::None;
  modeflags |= (perms & S_IRUSR) ? XrdCl::Access::UR : XrdCl::Access::None;
  modeflags |= (perms & S_IWUSR) ? XrdCl::Access::UW : XrdCl::Access::None;
  modeflags |= (perms & S_IXUSR) ? XrdCl::Access::UX : XrdCl::Access::None;
  modeflags |= (perms & S_IRGRP) ? XrdCl::Access::GR : XrdCl::Access::None;
  modeflags |= (perms & S_IWGRP) ? XrdCl::Access::GW : XrdCl::Access::None;
  modeflags |= (perms & S_IXGRP) ? XrdCl::Access::GX : XrdCl::Access::None;
  modeflags |= (perms & S_IROTH) ? XrdCl::Access::GR : XrdCl::Access::None;
  modeflags |= (perms & S_IWOTH) ? XrdCl::Access::GW : XrdCl::Access::None;
  modeflags |= (perms & S_IXOTH) ? XrdCl::Access::GX : XrdCl::Access::None;

  m_requestmanager = RequestManager::getInstance(name, openflags, modeflags);
  m_name = name;

  // Stat the file so we can keep track of the offset better.
  auto file = getActiveFile();
  XrdCl::XRootDStatus status;
  XrdCl::StatInfo *statInfo = nullptr;
  if (!(status = file->Stat(false, statInfo)).IsOK()) {
    edm::Exception ex(edm::errors::FileOpenError);
    ex << "XrdCl::File::Stat(name='" << name << ") => error '" << status.ToStr() << "' (errno=" << status.errNo
       << ", code=" << status.code << ")";
    ex.addContext("Calling XrdFile::open()");
    addConnection(ex);
    throw ex;
  }
  assert(statInfo);
  m_size = statInfo->GetSize();
  delete (statInfo);

  m_offset = 0;
  m_close = true;

  // Send the monitoring info, if available.
  // Note: std::getenv is not reentrant.
  // Commenting out until this is available in the new client.
  /*
  char * crabJobId = std::getenv(kCrabJobIdEnv);
  if (crabJobId) {
    kXR_unt32 dictId;
    m_file->SendMonitoringInfo(crabJobId, &dictId);
    edm::LogInfo("XrdFileInfo") << "Set monitoring ID to " << crabJobId << " with resulting dictId " << dictId << ".";
  }
*/

  edm::LogInfo("XrdFileInfo") << "Opened " << m_name;

  std::vector<std::string> sources;
  m_requestmanager->getActiveSourceNames(sources);
  std::stringstream ss;
  ss << "Active sources: ";
  for (auto const &it : sources)
    ss << it << ", ";
  edm::LogInfo("XrdFileInfo") << ss.str();
}

void XrdFile::close(void) {
  if (!m_requestmanager.get()) {
    edm::LogError("XrdFileError") << "XrdFile::close(name='" << m_name << "') called but the file is not open";
    m_close = false;
    return;
  }

  m_requestmanager = nullptr;  // propagate_const<T> has no reset() function

  m_close = false;
  m_offset = 0;
  m_size = -1;
  edm::LogInfo("XrdFileInfo") << "Closed " << m_name;
}

void XrdFile::abort(void) {
  m_requestmanager = nullptr;  // propagate_const<T> has no reset() function
  m_close = false;
  m_offset = 0;
  m_size = -1;
}

//////////////////////////////////////////////////////////////////////
IOSize XrdFile::read(void *into, IOSize n) {
  if (n > 0x7fffffff) {
    edm::Exception ex(edm::errors::FileReadError);
    ex << "XrdFile::read(name='" << m_name << "', n=" << n << ") too many bytes, limit is 0x7fffffff";
    ex.addContext("Calling XrdFile::read()");
    addConnection(ex);
    throw ex;
  }

  uint32_t bytesRead = m_requestmanager->handle(into, n, m_offset).get();
  m_offset += bytesRead;
  return bytesRead;
}

IOSize XrdFile::read(void *into, IOSize n, IOOffset pos) {
  if (n > 0x7fffffff) {
    edm::Exception ex(edm::errors::FileReadError);
    ex << "XrdFile::read(name='" << m_name << "', n=" << n << ") exceeds read size limit 0x7fffffff";
    ex.addContext("Calling XrdFile::read()");
    addConnection(ex);
    throw ex;
  }
  if (n == 0) {
    return 0;
  }

  // In some cases, the IO layers above us (particularly, if lazy-download is
  // enabled) will emit very large reads.  We break this up into multiple
  // reads in order to avoid hitting timeouts.
  std::future<IOSize> prev_future, cur_future;
  IOSize bytesRead = 0, prev_future_expected = 0, cur_future_expected = 0;
  bool readReturnedShort = false;

  // Check the status of a read operation; updates bytesRead and
  // readReturnedShort.
  auto check_read = [&](std::future<IOSize> &future, IOSize expected) {
    if (!future.valid()) {
      return;
    }
    IOSize result = future.get();
    if (readReturnedShort && (result != 0)) {
      edm::Exception ex(edm::errors::FileReadError);
      ex << "XrdFile::read(name='" << m_name << "', n=" << n
         << ") remote server returned non-zero length read after EOF.";
      ex.addContext("Calling XrdFile::read()");
      addConnection(ex);
      throw ex;
    } else if (result != expected) {
      readReturnedShort = true;
    }
    bytesRead += result;
  };

  while (n) {
    IOSize chunk = std::min(n, static_cast<IOSize>(XRD_CL_MAX_READ_SIZE));

    // Save prior read state; issue new read.
    prev_future = std::move(cur_future);
    prev_future_expected = cur_future_expected;
    cur_future = m_requestmanager->handle(into, chunk, pos);
    cur_future_expected = chunk;

    // Wait for the prior read; update bytesRead.
    check_read(prev_future, prev_future_expected);

    // Update counters.
    into = static_cast<char *>(into) + chunk;
    n -= chunk;
    pos += chunk;
  }

  // Wait for the last read to finish.
  check_read(cur_future, cur_future_expected);

  return bytesRead;
}

// This method is rarely used by CMS; hence, it is a small wrapper and not efficient.
IOSize XrdFile::readv(IOBuffer *into, IOSize n) {
  std::vector<IOPosBuffer> new_buf;
  new_buf.reserve(n);
  IOOffset off = 0;
  for (IOSize i = 0; i < n; i++) {
    IOSize size = into[i].size();
    new_buf[i] = IOPosBuffer(off, into[i].data(), size);
    off += size;
  }
  return readv(&(new_buf[0]), n);
}

/*
 * A vectored scatter-gather read.
 * Returns the total number of bytes successfully read.
 */
IOSize XrdFile::readv(IOPosBuffer *into, IOSize n) {
  // A trivial vector read - unlikely, considering ROOT data format.
  if (UNLIKELY(n == 0)) {
    return 0;
  }
  if (UNLIKELY(n == 1)) {
    return read(into[0].data(), into[0].size(), into[0].offset());
  }

  auto cl = std::make_shared<std::vector<IOPosBuffer>>();

  // CMSSW may issue large readv's; Xrootd is only able to handle
  // 1024.  Further, the splitting algorithm may slightly increase
  // the number of buffers.
  IOSize adjust = XRD_CL_MAX_SIZE - 2;
  cl->reserve(n > adjust ? adjust : n);
  IOSize idx = 0, last_idx = 0;
  IOSize final_result = 0;
  std::vector<std::pair<std::future<IOSize>, IOSize>> readv_futures;
  while (idx < n) {
    cl->clear();
    IOSize size = 0;
    while (idx < n) {
      unsigned rollback_count = 1;
      IOSize current_size = size;
      IOOffset offset = into[idx].offset();
      IOSize length = into[idx].size();
      size += length;
      char *buffer = static_cast<char *>(into[idx].data());
      while (length > XRD_CL_MAX_CHUNK) {
        IOPosBuffer ci;
        ci.set_size(XRD_CL_MAX_CHUNK);
        length -= XRD_CL_MAX_CHUNK;
        ci.set_offset(offset);
        offset += XRD_CL_MAX_CHUNK;
        ci.set_data(buffer);
        buffer += XRD_CL_MAX_CHUNK;
        cl->emplace_back(ci);
        rollback_count++;
      }
      IOPosBuffer ci;
      ci.set_size(length);
      ci.set_offset(offset);
      ci.set_data(buffer);
      cl->emplace_back(ci);

      if (cl->size() > adjust) {
        while (rollback_count--)
          cl->pop_back();
        size = current_size;
        break;
      } else {
        idx++;
      }
    }
    try {
      readv_futures.emplace_back(m_requestmanager->handle(cl), size);
    } catch (edm::Exception &ex) {
      ex.addContext("Calling XrdFile::readv()");
      throw;
    }

    // Assure that we have made some progress.
    assert(last_idx < idx);
    last_idx = idx;
  }
  std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
  start = std::chrono::high_resolution_clock::now();

  // If there are multiple readv calls, wait until all return until looking
  // at the results of any.  This guarantees that all readv's have finished
  // by time we call .get() for the first time (in case one of the readv's
  // result in an exception).
  //
  // We cannot have outstanding readv's on function exit as the XrdCl may
  // write into the corresponding buffer at the same time as ROOT.
  if (readv_futures.size() > 1) {
    for (auto &readv_result : readv_futures) {
      if (readv_result.first.valid()) {
        readv_result.first.wait();
      }
    }
  }

  for (auto &readv_result : readv_futures) {
    IOSize result = 0;
    try {
      const int retry_count = 5;
      for (int retries = 0; retries < retry_count; retries++) {
        try {
          if (readv_result.first.valid()) {
            result = readv_result.first.get();
          }
        } catch (XrootdException &ex) {
          if ((retries != retry_count - 1) && (ex.getCode() == XrdCl::errInvalidResponse)) {
            edm::LogWarning("XrdAdaptorInternal")
                << "Got an invalid response from Xrootd server; retrying" << std::endl;
            result = m_requestmanager->handle(cl).get();
          } else {
            throw;
          }
        }
        assert(result == readv_result.second);
      }
    } catch (edm::Exception &ex) {
      ex.addContext("Calling XrdFile::readv()");
      throw;
    } catch (std::exception &ex) {
      edm::Exception newex(edm::errors::StdException);
      newex << "A std::exception was thrown when processing an xrootd request: " << ex.what();
      newex.addContext("Calling XrdFile::readv()");
      throw newex;
    }
    final_result += result;
  }
  end = std::chrono::high_resolution_clock::now();

  edm::LogVerbatim("XrdAdaptorInternal")
      << "[" << m_op_count.fetch_add(1) << "] Time for readv: "
      << static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count())
      << " (sub-readv requests: " << readv_futures.size() << ")" << std::endl;

  return final_result;
}

IOSize XrdFile::write(const void *from, IOSize n) {
  if (n > 0x7fffffff) {
    edm::Exception ex(edm::errors::FileWriteError);
    ex << "XrdFile::write(name='" << m_name << "', n=" << n << ") too many bytes, limit is 0x7fffffff";
    ex.addContext("Calling XrdFile::write()");
    addConnection(ex);
    throw ex;
  }
  auto file = getActiveFile();

  XrdCl::XRootDStatus s = file->Write(m_offset, n, from);
  if (!s.IsOK()) {
    edm::Exception ex(edm::errors::FileWriteError);
    ex << "XrdFile::write(name='" << m_name << "', n=" << n << ") failed with error '" << s.ToStr()
       << "' (errno=" << s.errNo << ", code=" << s.code << ")";
    ex.addContext("Calling XrdFile::write()");
    addConnection(ex);
    throw ex;
  }
  m_offset += n;
  assert(m_size != -1);
  if (m_offset > m_size)
    m_size = m_offset;

  return n;
}

IOSize XrdFile::write(const void *from, IOSize n, IOOffset pos) {
  if (n > 0x7fffffff) {
    edm::Exception ex(edm::errors::FileWriteError);
    ex << "XrdFile::write(name='" << m_name << "', n=" << n << ") too many bytes, limit is 0x7fffffff";
    ex.addContext("Calling XrdFile::write()");
    addConnection(ex);
    throw ex;
  }
  auto file = getActiveFile();

  XrdCl::XRootDStatus s = file->Write(pos, n, from);
  if (!s.IsOK()) {
    edm::Exception ex(edm::errors::FileWriteError);
    ex << "XrdFile::write(name='" << m_name << "', n=" << n << ") failed with error '" << s.ToStr()
       << "' (errno=" << s.errNo << ", code=" << s.code << ")";
    ex.addContext("Calling XrdFile::write()");
    addConnection(ex);
    throw ex;
  }
  assert(m_size != -1);
  if (static_cast<IOOffset>(pos + n) > m_size)
    m_size = pos + n;

  return n;
}

bool XrdFile::prefetch(const IOPosBuffer *what, IOSize n) {
  // The new Xrootd client does not contain any internal buffers.
  // Hence, prefetching is disabled completely.
  return false;
}

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
IOOffset XrdFile::position(IOOffset offset, Relative whence /* = SET */) {
  if (!m_requestmanager.get()) {
    cms::Exception ex("FilePositionError");
    ex << "XrdFile::position() called on a closed file";
    ex.addContext("Calling XrdFile::position()");
    addConnection(ex);
    throw ex;
  }
  switch (whence) {
    case SET:
      m_offset = offset;
      break;

    case CURRENT:
      m_offset += offset;
      break;

    // TODO: None of this works with concurrent writers to the file.
    case END:
      assert(m_size != -1);
      m_offset = m_size + offset;
      break;

    default:
      cms::Exception ex("FilePositionError");
      ex << "XrdFile::position() called with incorrect 'whence' parameter";
      ex.addContext("Calling XrdFile::position()");
      addConnection(ex);
      throw ex;
  }

  if (m_offset < 0)
    m_offset = 0;
  assert(m_size != -1);
  if (m_offset > m_size)
    m_size = m_offset;

  return m_offset;
}

void XrdFile::resize(IOOffset /* size */) {
  cms::Exception ex("FileResizeError");
  ex << "XrdFile::resize(name='" << m_name << "') not implemented";
  ex.addContext("Calling XrdFile::resize()");
  addConnection(ex);
  throw ex;
}

std::shared_ptr<XrdCl::File> XrdFile::getActiveFile(void) {
  if (!m_requestmanager.get()) {
    cms::Exception ex("XrdFileLogicError");
    ex << "Xrd::getActiveFile(name='" << m_name << "') no active request manager";
    ex.addContext("Calling XrdFile::getActiveFile()");
    m_requestmanager->addConnections(ex);
    m_close = false;
    throw ex;
  }
  return m_requestmanager->getActiveFile();
}

void XrdFile::addConnection(cms::Exception &ex) {
  if (m_requestmanager.get()) {
    m_requestmanager->addConnections(ex);
  }
}
