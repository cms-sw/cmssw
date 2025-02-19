#ifndef STORAGE_FACTORY_LOCAL_CACHE_FILE_H
# define STORAGE_FACTORY_LOCAL_CACHE_FILE_H

# include "Utilities/StorageFactory/interface/Storage.h"
# include "Utilities/StorageFactory/interface/File.h"
# include <vector>
# include <string>

/** Proxy class to copy a file locally in large chunks. */
class LocalCacheFile : public Storage
{
public:
  LocalCacheFile (Storage *base, const std::string &tmpdir = "");
  ~LocalCacheFile (void);

  using Storage::read;
  using Storage::write;

  virtual bool		prefetch (const IOPosBuffer *what, IOSize n);
  virtual IOSize	read (void *into, IOSize n);
  virtual IOSize	read (void *into, IOSize n, IOOffset pos);
  virtual IOSize	readv (IOBuffer *into, IOSize n);
  virtual IOSize	readv (IOPosBuffer *into, IOSize n);
  virtual IOSize	write (const void *from, IOSize n);
  virtual IOSize	write (const void *from, IOSize n, IOOffset pos);
  virtual IOSize	writev (const IOBuffer *from, IOSize n);
  virtual IOSize	writev (const IOPosBuffer *from, IOSize n);

  virtual IOOffset	position (IOOffset offset, Relative whence = SET);
  virtual void		resize (IOOffset size);
  virtual void		flush (void);
  virtual void		close (void);

private:
  void			cache (IOOffset start, IOOffset end);

  IOOffset		image_;
  std::vector<char>	present_;
  File			*file_;
  Storage		*storage_;
  bool                  closedFile_;
  unsigned int          cacheCount_;
  unsigned int          cacheTotal_;
};

#endif // STORAGE_FACTORY_LOCAL_CACHE_FILE_H
