#ifndef Utilities_XrdAdaptor_XrdFile_h
#define Utilities_XrdAdaptor_XrdFile_h

#include "Utilities/StorageFactory/interface/Storage.h"
#include "Utilities/StorageFactory/interface/IOFlags.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/propagate_const.h"
#include "XrdCl/XrdClFile.hh"
#include <string>
#include <memory>
#include <atomic>

namespace XrdAdaptor {
  class RequestManager;
}

class XrdFile : public Storage {
public:
  XrdFile(void);
  XrdFile(IOFD fd);
  XrdFile(const char *name, int flags = IOFlags::OpenRead, int perms = 0666);
  XrdFile(const std::string &name, int flags = IOFlags::OpenRead, int perms = 0666);
  ~XrdFile(void) override;

  virtual void create(const char *name, bool exclusive = false, int perms = 0666);
  virtual void create(const std::string &name, bool exclusive = false, int perms = 0666);
  virtual void open(const char *name, int flags = IOFlags::OpenRead, int perms = 0666);
  virtual void open(const std::string &name, int flags = IOFlags::OpenRead, int perms = 0666);

  using Storage::position;
  using Storage::read;
  using Storage::readv;
  using Storage::write;

  bool prefetch(const IOPosBuffer *what, IOSize n) override;
  IOSize read(void *into, IOSize n) override;
  IOSize read(void *into, IOSize n, IOOffset pos) override;
  IOSize readv(IOBuffer *into, IOSize n) override;
  IOSize readv(IOPosBuffer *into, IOSize n) override;
  IOSize write(const void *from, IOSize n) override;
  IOSize write(const void *from, IOSize n, IOOffset pos) override;

  IOOffset position(IOOffset offset, Relative whence = SET) override;
  void resize(IOOffset size) override;

  void close(void) override;
  virtual void abort(void);

private:
  void addConnection(cms::Exception &);

  /**
   * Returns a file handle from one of the active sources.
   * Verifies the file is open and throws an exception as necessary.
   */
  std::shared_ptr<XrdCl::File> getActiveFile();

  edm::propagate_const<std::shared_ptr<XrdAdaptor::RequestManager>> m_requestmanager;
  IOOffset m_offset;
  IOOffset m_size;
  bool m_close;
  std::string m_name;
  std::atomic<unsigned int> m_op_count;
};

#endif  // XRD_ADAPTOR_XRD_FILE_H
