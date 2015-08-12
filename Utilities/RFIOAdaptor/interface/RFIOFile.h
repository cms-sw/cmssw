#ifndef RFIO_ADAPTOR_RFIO_FILE_H
#define RFIO_ADAPTOR_RFIO_FILE_H

# include "Utilities/StorageFactory/interface/Storage.h"
# include "Utilities/StorageFactory/interface/IOFlags.h"
# include <string>

/** RFIO #Storage object.  */
class RFIOFile final : public Storage
{
public:
  RFIOFile (void);
  RFIOFile (IOFD fd);
  RFIOFile (const char *name, int flags = IOFlags::OpenRead, int perms = 0666);
  RFIOFile (const std::string &name, int flags = IOFlags::OpenRead, int perms = 0666);
  ~RFIOFile (void);

   void		create (const char *name,
				bool exclusive = false,
				int perms = 0666) ;
   void		create (const std::string &name,
				bool exclusive = false,
				int perms = 0666) ;
   void		open (const char *name,
			      int flags = IOFlags::OpenRead,
			      int perms = 0666) ;
   void		open (const std::string &name,
			      int flags = IOFlags::OpenRead,
			      int perms = 0666) ;

  using Storage::read;
  using Storage::readv;
  using Storage::write;
  using Storage::position;

  virtual IOSize	read (void *into, IOSize n) override;
  virtual IOSize	readv (IOPosBuffer *into, IOSize buffers) override;
  virtual IOSize	write (const void *from, IOSize n) override;
  virtual IOOffset	position (IOOffset offset, Relative whence = SET) override;
  virtual void		resize (IOOffset size) override;
  virtual void		close (void) override;
  void		abort (void) ;

/*
 * Note: we used to implement prefetch for RFIOFile, but it never got used in
 * production due to memory leaks in the underlying library.  This was removed
 * in CMSSW 6 so we could default to storage-only if available.
 */

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
