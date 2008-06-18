#ifndef RFIO_ADAPTOR_RFIO_FILE_H
# define RFIO_ADAPTOR_RFIO_FILE_H

# include "Utilities/StorageFactory/interface/Storage.h"
# include "Utilities/StorageFactory/interface/IOChannel.h"
# include "Utilities/StorageFactory/interface/IOFlags.h"
# include <string>

/** RFIO #Storage object.  */
class RFIOFile : public Storage
{
public:
  RFIOFile (void);
  RFIOFile (IOFD fd);
  RFIOFile (const char *name, int flags = IOFlags::OpenRead, int perms = 0666);
  RFIOFile (const std::string &name, int flags = IOFlags::OpenRead, int perms = 0666);
  ~RFIOFile (void);

  virtual void		create (const char *name,
				bool exclusive = false,
				int perms = 0666);
  virtual void		create (const std::string &name,
				bool exclusive = false,
				int perms = 0666);
  virtual void		open (const char *name,
			      int flags = IOFlags::OpenRead,
			      int perms = 0666);
  virtual void		open (const std::string &name,
			      int flags = IOFlags::OpenRead,
			      int perms = 0666);

  using Storage::read;
  using Storage::readv;
  using Storage::write;
  using Storage::position;
  using Storage::prefetch;

  virtual bool		prefetch (const IOPosBuffer *what, IOSize n);
  virtual IOSize	read (void *into, IOSize n);
  virtual IOSize	readv (IOPosBuffer *into, IOSize buffers);
  virtual IOSize	write (const void *from, IOSize n);
  virtual IOOffset	position (IOOffset offset, Relative whence = SET);
  virtual void		resize (IOOffset size);
  virtual void		close (void);
  virtual void		abort (void);

private:
  ssize_t		retryRead (void *into, IOSize n, int maxRetry = 10);
  void			reopen();

  IOFD			m_fd;
  bool			m_close;
  std::string		m_name;
  int			m_flags;
  int			m_perms;
  IOOffset              m_curpos;
};

#endif // RFIO_ADAPTOR_RFIO_FILE_H
