#include "Utilities/StorageFactory/interface/LocalCacheFile.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include <utility>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include <errno.h>
#include <sstream>

static const IOOffset CHUNK_SIZE = 128*1024*1024;

static void
nowrite(const char *why)
{
  throw cms::Exception("LocalCacheFile")
    << "Cannot change file but operation '" << why << "' was called";
}


LocalCacheFile::LocalCacheFile(Storage *base, const std::string &tmpdir /* = "" */)
  : image_(base->size()),
    file_(0),
    storage_(base),
    closedFile_(false),
    cacheCount_(0),
    cacheTotal_((image_ + CHUNK_SIZE - 1) / CHUNK_SIZE)
{
  present_.resize(cacheTotal_, 0);

  std::string pattern(tmpdir);
  if (pattern.empty())
    if (char *p = getenv("TMPDIR"))
      pattern = p;
  if (pattern.empty())
    pattern = "/tmp";
  pattern += "/cmssw-shadow-XXXXXX";

  std::vector<char> temp(pattern.c_str(), pattern.c_str()+pattern.size()+1);
  int fd = mkstemp(&temp[0]);
  if (fd == -1)
    throw cms::Exception("LocalCacheFile")
      << "Cannot create temporary file '" << pattern << "': "
      << strerror(errno) << " (error " << errno << ")";

  unlink(&temp[0]);
  file_ = new File(fd);
  file_->resize(image_);
}

LocalCacheFile::~LocalCacheFile(void)
{
  delete file_;
  delete storage_;
}

void
LocalCacheFile::cache(IOOffset start, IOOffset end)
{
  start = (start / CHUNK_SIZE) * CHUNK_SIZE;
  end = std::min(end, image_);

  IOSize nread = 0;
  IOSize index = start / CHUNK_SIZE;

  while (start < end)
  {
    IOSize len = std::min(image_ - start, CHUNK_SIZE);
    if (! present_[index])
    {
      void *window = mmap(0, len, PROT_READ | PROT_WRITE, MAP_SHARED, file_->fd(), start);
      if (window == MAP_FAILED)
        throw cms::Exception("LocalCacheFile")
	  << "Unable to map a window of local cache file: "
	  << strerror(errno) << " (error " << errno << ")";

      try
      {
        nread = storage_->read(window, len, start);
      }
      catch (cms::Exception &e)
      {
        munmap(window, len);
	std::ostringstream ost;
        ost << "Unable to cache " << len << " byte file segment at " << start << ": ";
        throw cms::Exception("LocalCacheFile", ost.str(), e);
      }

      munmap(window, len);

      if (nread != len)
        throw cms::Exception("LocalCacheFile")
          << "Unable to cache " << len << " byte file segment at " << start
	  << ": got only " << nread << " bytes back";

      present_[index] = 1;
      ++cacheCount_;
      if (cacheCount_ == cacheTotal_)
      {
        storage_->close();
        closedFile_ = true;
      }
    }

    start += len;
    ++index;
  }
}

IOSize
LocalCacheFile::read(void *into, IOSize n)
{
  IOOffset here = file_->position();
  cache(here, here + n);

  return file_->read(into, n);
}

IOSize
LocalCacheFile::read(void *into, IOSize n, IOOffset pos)
{
  cache(pos, pos + n);
  return file_->read(into, n, pos);
}

IOSize
LocalCacheFile::readv(IOBuffer *into, IOSize n)
{
  IOOffset start = file_->position();
  IOOffset end = start;
  for (IOSize i = 0; i < n; ++i)
    end += into[i].size();
  cache(start, end);

  return file_->readv(into, n);
}

IOSize
LocalCacheFile::readv(IOPosBuffer *into, IOSize n)
{
  for (IOSize i = 0; i < n; ++i)
  {
    IOOffset start = into[i].offset();
    IOOffset end = start + into[i].size();
    cache(start, end);
  }

  return storage_->readv(into, n);
}

IOSize
LocalCacheFile::write(const void */*from*/, IOSize)
{ nowrite("write"); return 0; }

IOSize
LocalCacheFile::write(const void */*from*/, IOSize, IOOffset /*pos*/)
{ nowrite("write"); return 0; }

IOSize
LocalCacheFile::writev(const IOBuffer */*from*/, IOSize)
{ nowrite("writev"); return 0; }

IOSize
LocalCacheFile::writev(const IOPosBuffer */*from*/, IOSize)
{ nowrite("writev"); return 0; }

IOOffset
LocalCacheFile::position(IOOffset offset, Relative whence)
{ return file_->position(offset, whence); }

void
LocalCacheFile::resize(IOOffset /*size*/)
{ nowrite("resize"); }

void
LocalCacheFile::flush(void)
{ nowrite("flush"); }

void
LocalCacheFile::close(void)
{
  if (!closedFile_)
  {
    storage_->close();
  }
  file_->close();
}

bool
LocalCacheFile::prefetch(const IOPosBuffer *what, IOSize n)
{
  for (IOSize i = 0; i < n; ++i)
  {
    IOOffset start = what[i].offset();
    IOOffset end = start + what[i].size();
    cache(start, end);
  }
  
  return file_->prefetch(what, n);
}
