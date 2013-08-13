#include "Utilities/XrdAdaptor/src/XrdFile.h"
#include "Utilities/XrdAdaptor/src/XrdRequestManager.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Likely.h"
#include "FWCore/Utilities/interface/CPUTimer.h"
#include <vector>
#include <sstream>
#include <iostream>
#include <assert.h>

using namespace XrdAdaptor;

// To be re-enabled when the monitoring interface is back.
//static const char *kCrabJobIdEnv = "CRAB_UNIQUE_JOB_ID";

#define XRD_CL_MAX_CHUNK 512*1024

XrdFile::XrdFile (void)
  :  m_offset (0),
    m_size(-1),
    m_close (false),
    m_name(),
    m_op_count(0)
{
}

XrdFile::XrdFile (const char *name,
    	          int flags /* = IOFlags::OpenRead */,
    	          int perms /* = 066 */)
  : m_offset (0),
    m_size(-1),
    m_close (false),
    m_name(),
    m_op_count(0)
{
  open (name, flags, perms);
}

XrdFile::XrdFile (const std::string &name,
    	          int flags /* = IOFlags::OpenRead */,
    	          int perms /* = 066 */)
  : m_offset (0),
    m_size(-1),
    m_close (false),
    m_name(),
    m_op_count(0)
{
  open (name.c_str (), flags, perms);
}

XrdFile::~XrdFile (void)
{
  if (m_close)
    edm::LogError("XrdFileError")
      << "Destructor called on XROOTD file '" << m_name
      << "' but the file is still open";
}

//////////////////////////////////////////////////////////////////////
void
XrdFile::create (const char *name,
		 bool exclusive /* = false */,
		 int perms /* = 066 */)
{
  open (name,
        (IOFlags::OpenCreate | IOFlags::OpenWrite | IOFlags::OpenTruncate
         | (exclusive ? IOFlags::OpenExclusive : 0)),
        perms);
}

void
XrdFile::create (const std::string &name,
                 bool exclusive /* = false */,
                 int perms /* = 066 */)
{
  open (name.c_str (),
        (IOFlags::OpenCreate | IOFlags::OpenWrite | IOFlags::OpenTruncate
         | (exclusive ? IOFlags::OpenExclusive : 0)),
        perms);
}

void
XrdFile::open (const std::string &name,
               int flags /* = IOFlags::OpenRead */,
               int perms /* = 066 */)
{ open (name.c_str (), flags, perms); }

void
XrdFile::open (const char *name,
               int flags /* = IOFlags::OpenRead */,
               int perms /* = 066 */)
{
  // Actual open
  if ((name == 0) || (*name == 0)) {
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

  if (flags & IOFlags::OpenCreate)
  {
    if (! (flags & IOFlags::OpenExclusive))
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

  m_requestmanager.reset(new RequestManager(name, openflags, modeflags));
  m_name = name;

  // Stat the file so we can keep track of the offset better.
  auto file = getActiveFile();
  XrdCl::XRootDStatus status;
  XrdCl::StatInfo *statInfo = NULL;
  if (! (status = file->Stat(true, statInfo)).IsOK()) {
    edm::Exception ex(edm::errors::FileOpenError);
    ex << "XrdCl::File::Stat(name='" << name
       << ") => error '" << status.ToString()
       << "' (errno=" << status.errNo << ", code=" << status.code << ")";
    ex.addContext("Calling XrdFile::open()");
    addConnection(ex);
    throw ex;
  }
  assert(statInfo);
  m_size = statInfo->GetSize();
  delete(statInfo);

  m_offset = 0;
  m_close = true;

  // Send the monitoring info, if available.
  // Note: getenv is not reentrant.
  // Commenting out until this is available in the new client.
/*
  char * crabJobId = getenv(kCrabJobIdEnv);
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
  for (auto const& it : sources)
    ss << it << ", ";
  edm::LogInfo("XrdFileInfo") << ss.str();
}

void
XrdFile::close (void)
{
  if (! m_requestmanager.get())
  {
    edm::LogError("XrdFileError")
      << "XrdFile::close(name='" << m_name
      << "') called but the file is not open";
    m_close = false;
    return;
  }

  m_requestmanager.reset();

  m_close = false;
  m_offset = 0;
  m_size = -1;
  edm::LogInfo("XrdFileInfo") << "Closed " << m_name;
}

void
XrdFile::abort (void)
{
  m_requestmanager.reset(nullptr);
  m_close = false;
  m_offset = 0;
  m_size = -1;
}

//////////////////////////////////////////////////////////////////////
IOSize
XrdFile::read (void *into, IOSize n)
{
  if (n > 0x7fffffff) {
    edm::Exception ex(edm::errors::FileReadError);
    ex << "XrdFile::read(name='" << m_name << "', n=" << n
       << ") too many bytes, limit is 0x7fffffff";
    ex.addContext("Calling XrdFile::read()");
    addConnection(ex);
    throw ex;
  }

  uint32_t bytesRead = m_requestmanager->handle(into, n, m_offset).get();
  m_offset += bytesRead;
  return bytesRead;
}

IOSize
XrdFile::read (void *into, IOSize n, IOOffset pos)
{
  if (n > 0x7fffffff) {
    edm::Exception ex(edm::errors::FileReadError);
    ex << "XrdFile::read(name='" << m_name << "', n=" << n
       << ") exceeds read size limit 0x7fffffff";
    ex.addContext("Calling XrdFile::read()");
    addConnection(ex);
    throw ex;
  }

  uint32_t bytesRead = m_requestmanager->handle(into, n, pos).get();

  return bytesRead;
}

// This method is rarely used by CMS; hence, it is a small wrapper and not efficient.
IOSize
XrdFile::readv (IOBuffer *into, IOSize n)
{
  std::vector<IOPosBuffer> new_buf;
  new_buf.reserve(n);
  IOOffset off = 0;
  for (IOSize i=0; i<n; i++) {
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
IOSize
XrdFile::readv (IOPosBuffer *into, IOSize n)
{
  // A trivial vector read - unlikely, considering ROOT data format.
  if (unlikely(n == 0)) {
    return 0;
  }
  if (unlikely(n == 1)) {
    return read(into[0].data(), into[0].size(), into[0].offset());
  }

  std::shared_ptr<std::vector<IOPosBuffer> >cl(new std::vector<IOPosBuffer>);
  cl->reserve(n);

  IOSize size = 0;
  for (IOSize i=0; i<n; i++) {
    IOOffset offset = into[i].offset();
    IOSize length = into[i].size();
    size += length;
    char * buffer = static_cast<char *>(into[i].data());
    while (length > XRD_CL_MAX_CHUNK) {
      IOPosBuffer ci;
      ci.set_size(XRD_CL_MAX_CHUNK);
      length -= XRD_CL_MAX_CHUNK;
      ci.set_offset(offset);
      offset += XRD_CL_MAX_CHUNK;
      ci.set_data(buffer);
      buffer += XRD_CL_MAX_CHUNK;
      cl->emplace_back(ci);
    }
    IOPosBuffer ci;
    ci.set_size(length);
    ci.set_offset(offset);
    ci.set_data(buffer);
    cl->emplace_back(ci);
  }
  edm::CPUTimer timer;
  timer.start();
  IOSize result;
  try
  {
    result = m_requestmanager->handle(cl).get();
  }
  catch (edm::Exception& ex)
  {
    ex.addContext("Calling XrdFile::readv()");
    throw;
  }
  timer.stop();
  assert(result == size);
  edm::LogVerbatim("XrdAdaptorInternal") << "[" << m_op_count.fetch_add(1) << "] Time for readv: " << static_cast<int>(1000*timer.realTime()) << std::endl;
  return result;
}

IOSize
XrdFile::write (const void *from, IOSize n)
{
  if (n > 0x7fffffff) {
    cms::Exception ex("FileWriteError");
    ex << "XrdFile::write(name='" << m_name << "', n=" << n
       << ") too many bytes, limit is 0x7fffffff";
    ex.addContext("Calling XrdFile::write()");
    addConnection(ex);
    throw ex;
  }
  auto file = getActiveFile();

  XrdCl::XRootDStatus s = file->Write(m_offset, n, from);
  if (!s.IsOK()) {
    cms::Exception ex("FileWriteError");
    ex << "XrdFile::write(name='" << m_name << "', n=" << n
       << ") failed with error '" << s.ToString()
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

IOSize
XrdFile::write (const void *from, IOSize n, IOOffset pos)
{
  if (n > 0x7fffffff) {
    cms::Exception ex("FileWriteError");
    ex << "XrdFile::write(name='" << m_name << "', n=" << n
       << ") too many bytes, limit is 0x7fffffff";
    ex.addContext("Calling XrdFile::write()");
    addConnection(ex);
    throw ex;
  }
  auto file = getActiveFile();

  XrdCl::XRootDStatus s = file->Write(pos, n, from);
  if (!s.IsOK()) {
    cms::Exception ex("FileWriteError");
    ex << "XrdFile::write(name='" << m_name << "', n=" << n
       << ") failed with error '" << s.ToString()
       << "' (errno=" << s.errNo << ", code=" << s.code << ")";
    ex.addContext("Calling XrdFile::write()");
    addConnection(ex);
    throw ex;
  }
  assert (m_size != -1);
  if (static_cast<IOOffset>(pos + n) > m_size)
    m_size = pos + n;

  return n;
}

bool
XrdFile::prefetch (const IOPosBuffer *what, IOSize n)
{
  // The new Xrootd client does not contain any internal buffers.
  // Hence, prefetching is disabled completely.
  return false;
}

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
IOOffset
XrdFile::position (IOOffset offset, Relative whence /* = SET */)
{
  if (! m_requestmanager.get()) {
    cms::Exception ex("FilePositionError");
    ex << "XrdFile::position() called on a closed file";
    ex.addContext("Calling XrdFile::position()");
    addConnection(ex);
    throw ex;
  }
  switch (whence)
  {
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

void
XrdFile::resize (IOOffset /* size */)
{
  cms::Exception ex("FileResizeError");
  ex << "XrdFile::resize(name='" << m_name << "') not implemented";
  ex.addContext("Calling XrdFile::resize()");
  addConnection(ex);
  throw ex;
}

std::shared_ptr<XrdCl::File>
XrdFile::getActiveFile (void) 
{ 
  if (!m_requestmanager.get())
  { 
    cms::Exception ex("XrdFileLogicError");
    ex << "Xrd::getActiveFile(name='" << m_name << "') no active request manager";
    ex.addContext("Calling XrdFile::getActiveFile()");
    m_requestmanager->addConnections(ex);
    m_close = false;
    throw ex;
  }
  return m_requestmanager->getActiveFile();
}

void
XrdFile::addConnection (cms::Exception &ex)
{
  if (m_requestmanager.get())
  {
    m_requestmanager->addConnections(ex);
  }
}

