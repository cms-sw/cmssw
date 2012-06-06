#include "Utilities/XrdAdaptor/src/XrdFile.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <vector>
#include <sstream>

static const char *kCrabJobIdEnv = "CRAB_UNIQUE_JOB_ID";

XrdFile::XrdFile (void)
  : m_client (0),
    m_offset (0),
    m_stat(),
    m_close (false),
    m_name()
{
  memset(&m_stat, 0, sizeof (m_stat));
}

XrdFile::XrdFile (const char *name,
    	          int flags /* = IOFlags::OpenRead */,
    	          int perms /* = 066 */)
  : m_client (0),
    m_offset (0),
    m_stat(),
    m_close (false),
    m_name()
{
  memset(&m_stat, 0, sizeof (m_stat));
  open (name, flags, perms);
}

XrdFile::XrdFile (const std::string &name,
    	          int flags /* = IOFlags::OpenRead */,
    	          int perms /* = 066 */)
  : m_client (0),
    m_offset (0),
    m_stat(),
    m_close (false),
    m_name()
{
  memset(&m_stat, 0, sizeof (m_stat));
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
  // If I am already open, close old file first
  if (m_client && m_close)
    close();
  else
    abort();

  // Translate our flags to system flags
  int openflags = 0;

  if (flags & IOFlags::OpenWrite)
    openflags |= kXR_open_updt;
  else if (flags & IOFlags::OpenRead)
    openflags |= kXR_open_read;

  if (flags & IOFlags::OpenAppend) {
    edm::Exception ex(edm::errors::FileOpenError);
    ex << "Opening file '" << name << "' in append mode not supported";
    ex.addContext("Calling XrdFile::open()");
    throw ex;
  }

  if (flags & IOFlags::OpenCreate)
  {
    if (! (flags & IOFlags::OpenExclusive))
      openflags |= kXR_delete;
    openflags |= kXR_new;
    openflags |= kXR_mkpath;
  }

  if ((flags & IOFlags::OpenTruncate) && (flags & IOFlags::OpenWrite))
    openflags |= kXR_delete;

  m_name = name;
  m_client = new XrdClient(name);
  if (! m_client->Open(perms, openflags)
      || m_client->LastServerResp()->status != kXR_ok) {
    edm::Exception ex(edm::errors::FileOpenError);
    ex << "XrdClient::Open(name='" << name
       << "', flags=0x" << std::hex << openflags
       << ", permissions=0" << std::oct << perms << std::dec
       << ") => error '" << m_client->LastServerError()->errmsg
       << "' (errno=" << m_client->LastServerError()->errnum << ")";
    ex.addContext("Calling XrdFile::open()");
    addConnection(ex);
    throw ex;
  }
  if (! m_client->Stat(&m_stat)) {
    edm::Exception ex(edm::errors::FileOpenError);
    ex << "XrdClient::Stat(name='" << name
      << ") => error '" << m_client->LastServerError()->errmsg
      << "' (errno=" << m_client->LastServerError()->errnum << ")";
    ex.addContext("Calling XrdFile::open()");
    addConnection(ex);
    throw ex;
  }
  m_offset = 0;
  m_close = true;

  // Send the monitoring info, if available.
  // Note: getenv is not reentrant.
  char * crabJobId = getenv(kCrabJobIdEnv);
  if (crabJobId) {
    kXR_unt32 dictId;
    m_client->SendMonitoringInfo(crabJobId, &dictId);
    edm::LogInfo("XrdFileInfo") << "Set monitoring ID to " << crabJobId << " with resulting dictId " << dictId << ".";
  }

  edm::LogInfo("XrdFileInfo") << "Opened " << m_name;

  XrdClientConn *conn = m_client->GetClientConn();
  edm::LogInfo("XrdFileInfo") << "Connection URL " << conn->GetCurrentUrl().GetUrl().c_str();
}

void
XrdFile::close (void)
{
  if (! m_client)
  {
    edm::LogError("XrdFileError")
      << "XrdFile::close(name='" << m_name
      << "') called but the file is not open";
    m_close = false;
    return;
  }

  if (! m_client->Close())
    edm::LogWarning("XrdFileWarning")
      << "XrdFile::close(name='" << m_name
      << "') failed with error '" << m_client->LastServerError()->errmsg
      << "' (errno=" << m_client->LastServerError()->errnum << ")";
  delete m_client;
  m_client = 0;

  m_close = false;
  m_offset = 0;
  memset(&m_stat, 0, sizeof (m_stat));
  edm::LogInfo("XrdFileInfo") << "Closed " << m_name;
}

void
XrdFile::abort (void)
{
  delete m_client;
  m_client = 0;
  m_close = false;
  m_offset = 0;
  memset(&m_stat, 0, sizeof (m_stat));
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
  int s = m_client->Read(into, m_offset, n);
  if (s < 0) {
    edm::Exception ex(edm::errors::FileReadError);
    ex << "XrdClient::Read(name='" << m_name
       << "', offset=" << m_offset << ", n=" << n
       << ") failed with error '" << m_client->LastServerError()->errmsg
       << "' (errno=" << m_client->LastServerError()->errnum << ")";
    ex.addContext("Calling XrdFile::read()");
    addConnection(ex);
    throw ex;
  }
  m_offset += s;
  return s;
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
  int s = m_client->Read(into, pos, n);
  if (s < 0) {
    edm::Exception ex(edm::errors::FileReadError);
    ex << "XrdClient::Read(name='" << m_name
       << "', offset=" << m_offset << ", n=" << n
       << ") failed with error '" << m_client->LastServerError()->errmsg
       << "' (errno=" << m_client->LastServerError()->errnum << ")";
    ex.addContext("Calling XrdFile::read()");
    addConnection(ex);
    throw ex;
  }
  return s;
}

IOSize
XrdFile::readv (IOBuffer *into, IOSize n)
{
  // Note that the XROOTD and our interfaces do not match.
  // XROOTD expects readv() into a single monolithic buffer
  // from multiple file locations, whereas we expect scatter
  // gather buffers read from a single file location.
  //
  // We work around this mismatch for now by issuing a cache
  // read on the background, and then individual reads.
  IOOffset pos = m_offset;
  std::vector<long long> offsets (n);
  std::vector<int> lengths (n);
  for (IOSize i = 0; i < n; ++i)
  {
    IOSize len = into[i].size();
    if (len > 0x7fffffff) {
      edm::Exception ex(edm::errors::FileReadError);
      ex << "XrdFile::readv(name='" << m_name << "')[" << i
         << "].size=" << len << " exceeds read size limit 0x7fffffff";
      ex.addContext("Calling XrdFile::readv()");
      addConnection(ex);
      throw ex;
    }
    offsets[i] = pos;
    lengths[i] = len;
    pos += len;
  }

  // Prefetch into the cache (if any).
  if (m_client->ReadV(0, &offsets[0], &lengths[0], n) < 0) {
    edm::Exception ex(edm::errors::FileReadError);
    ex << "XrdClient::ReadV(name='" << m_name
       << "') failed with error '" << m_client->LastServerError()->errmsg
       << "' (errno=" << m_client->LastServerError()->errnum << ")";
    ex.addContext("Calling XrdFile::readv()");
    addConnection(ex);
    throw ex;
  }

  // Issue actual reads.
  IOSize total = 0;
  for (IOSize i = 0; i < n; ++i)
  {
    int s = m_client->Read(into[i].data(), offsets[i], lengths[i]);
    if (s < 0)
    {
      if (i > 0)
        break;
      edm::Exception ex(edm::errors::FileReadError);
      ex << "XrdClient::Read(name='" << m_name
         << "', offset=" << offsets[i] << ", n=" << lengths[i]
         << ") failed with error '" << m_client->LastServerError()->errmsg
         << "' (errno=" << m_client->LastServerError()->errnum << ")";
      ex.addContext("Calling XrdFile::readv()");
      addConnection(ex);
      throw ex;
    }
    total += s;
    m_offset += s;
  }

  return total;
}

IOSize
XrdFile::readv (IOPosBuffer *into, IOSize n)
{
  // See comments in readv() above.
  std::vector<long long> offsets (n);
  std::vector<int> lengths (n);
  for (IOSize i = 0; i < n; ++i)
  {
    IOSize len = into[i].size();
    if (len > 0x7fffffff) {
      edm::Exception ex(edm::errors::FileReadError);
      ex << "XrdFile::readv(name='" << m_name << "')[" << i
         << "].size=" << len << " exceeds read size limit 0x7fffffff";
      ex.addContext("Calling XrdFile::readv()");
      addConnection(ex);
      throw ex;
    }
    offsets[i] = into[i].offset();
    lengths[i] = len;
  }

  // Prefetch into the cache (if any).
  if (m_client->ReadV(0, &offsets[0], &lengths[0], n) < 0) {
    edm::Exception ex(edm::errors::FileReadError);
    ex << "XrdClient::ReadV(name='" << m_name
       << "') failed with error '" << m_client->LastServerError()->errmsg
       << "' (errno=" << m_client->LastServerError()->errnum << ")";
    ex.addContext("Calling XrdFile::readv()");
    addConnection(ex);
    throw ex;
  }
  // Issue actual reads.
  IOSize total = 0;
  for (IOSize i = 0; i < n; ++i)
  {
    int s = m_client->Read(into[i].data(), offsets[i], lengths[i]);
    if (s < 0)
    {
      if (i > 0)
        break;
      edm::Exception ex(edm::errors::FileReadError);
      ex << "XrdClient::Read(name='" << m_name
         << "', offset=" << offsets[i] << ", n=" << lengths[i]
         << ") failed with error '" << m_client->LastServerError()->errmsg
         << "' (errno=" << m_client->LastServerError()->errnum << ")";
      ex.addContext("Calling XrdFile::readv()");
      addConnection(ex);
      throw ex;
    }
    total += s;
  }

  return total;
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
  ssize_t s = m_client->Write(from, m_offset, n);
  if (s < 0) {
    cms::Exception ex("FileWriteError");
    ex << "XrdFile::write(name='" << m_name << "', n=" << n
       << ") failed with error '" << m_client->LastServerError()->errmsg
       << "' (errno=" << m_client->LastServerError()->errnum << ")";
    ex.addContext("Calling XrdFile::write()");
    addConnection(ex);
    throw ex;
  }
  m_offset += s;
  if (m_offset > m_stat.size)
    m_stat.size = m_offset;

  return s;
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
  ssize_t s = m_client->Write(from, pos, n);
  if (s < 0) {
    cms::Exception ex("FileWriteError");
    ex << "XrdFile::write(name='" << m_name << "', n=" << n
       << ") failed with error '" << m_client->LastServerError()->errmsg
       << "' (errno=" << m_client->LastServerError()->errnum << ")";
    ex.addContext("Calling XrdFile::write()");
    addConnection(ex);
    throw ex;
  }
  if (pos + s > m_stat.size)
    m_stat.size = pos + s;

  return s;
}

bool
XrdFile::prefetch (const IOPosBuffer *what, IOSize n)
{
  std::vector<long long> offsets; offsets.resize(n);
  std::vector<int> lens; lens.resize(n);
  kXR_int64 total = 0;
  for (IOSize i = 0; i < n; ++i) {
    offsets[i] = what[i].offset();
    lens[i] = what[i].size();
    total += what[i].size();
  }

  kXR_int64 r = m_client->ReadV(NULL, &offsets[0], &lens[0], n);
  return r == total;
}

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
IOOffset
XrdFile::position (IOOffset offset, Relative whence /* = SET */)
{
  if (! m_client) {
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

  case END:
    m_offset = m_stat.size + offset;
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
  if (m_offset > m_stat.size)
    m_stat.size = m_offset;

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

void
XrdFile::addConnection (cms::Exception &ex)
{
  XrdClientConn *conn = m_client->GetClientConn();
  if (conn) {
    std::stringstream ss;
    ss << "Current server connection: " << conn->GetCurrentUrl().GetUrl().c_str();
    ex.addAdditionalInfo(ss.str());
  }
}

