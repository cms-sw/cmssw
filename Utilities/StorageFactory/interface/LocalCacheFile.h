#ifndef STORAGE_FACTORY_LOCAL_CACHE_FILE_H
#define STORAGE_FACTORY_LOCAL_CACHE_FILE_H

#include "Utilities/StorageFactory/interface/Storage.h"
#include "Utilities/StorageFactory/interface/File.h"
#include "FWCore/Utilities/interface/propagate_const.h"
#include <vector>
#include <string>
#include <memory>

/** Proxy class to copy a file locally in large chunks. */
class LocalCacheFile : public Storage {
public:
  LocalCacheFile(std::unique_ptr<Storage> base, const std::string &tmpdir = "");
  ~LocalCacheFile(void) override;

  using Storage::read;
  using Storage::write;

  bool prefetch(const IOPosBuffer *what, IOSize n) override;
  IOSize read(void *into, IOSize n) override;
  IOSize read(void *into, IOSize n, IOOffset pos) override;
  IOSize readv(IOBuffer *into, IOSize n) override;
  IOSize readv(IOPosBuffer *into, IOSize n) override;
  IOSize write(const void *from, IOSize n) override;
  IOSize write(const void *from, IOSize n, IOOffset pos) override;
  IOSize writev(const IOBuffer *from, IOSize n) override;
  IOSize writev(const IOPosBuffer *from, IOSize n) override;

  IOOffset position(IOOffset offset, Relative whence = SET) override;
  void resize(IOOffset size) override;
  void flush(void) override;
  void close(void) override;

private:
  void cache(IOOffset start, IOOffset end);

  IOOffset image_;
  std::vector<char> present_;
  edm::propagate_const<std::unique_ptr<File>> file_;
  edm::propagate_const<std::unique_ptr<Storage>> storage_;
  bool closedFile_;
  unsigned int cacheCount_;
  unsigned int cacheTotal_;
};

#endif  // STORAGE_FACTORY_LOCAL_CACHE_FILE_H
