#ifndef STORAGE_FACTORY_FILE_H
# define STORAGE_FACTORY_FILE_H

# include "Utilities/StorageFactory/interface/IOTypes.h"
# include "Utilities/StorageFactory/interface/IOFlags.h"
# include "Utilities/StorageFactory/interface/Storage.h"
# include "Utilities/StorageFactory/interface/IOChannel.h"
# include <string>

/** Basic file-related functions.  Nicked from SEAL.  */
class File : public IOChannel, public Storage
{
public:
  File (void);
  File (IOFD fd, bool autoclose = true);
  File (const char *name, int flags = IOFlags::OpenRead, int perms = 0666);
  File (const std::string &name, int flags = IOFlags::OpenRead, int perms = 0666);
  ~File (void);
  // implicit copy constructor
  // implicit assignment operator

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
  virtual void		attach (IOFD fd);

  using Storage::read;
  using Storage::write;
  using Storage::position;

  virtual bool		prefetch (const IOPosBuffer *what, IOSize n);
  virtual IOSize	read (void *into, IOSize n);
  virtual IOSize	read (void *into, IOSize n, IOOffset pos);
  virtual IOSize	readv (IOBuffer *into, IOSize length);

  virtual IOSize	write (const void *from, IOSize n);
  virtual IOSize	write (const void *from, IOSize n, IOOffset pos);
  virtual IOSize	writev (const IOBuffer *from, IOSize length);

  virtual IOOffset	size (void) const;
  virtual IOOffset	position (IOOffset offset, Relative whence = SET);

  virtual void		resize (IOOffset size);

  virtual void		flush (void);
  virtual void		close (void);
  virtual void		abort (void);

  virtual void		setAutoClose (bool closeit);

private:
  enum { InternalAutoClose = 4096 }; //< Close on delete

  File (IOFD fd, unsigned flags);

  File *		duplicate (bool copy) const;
  File *		duplicate (File *child) const;
  static IOFD		sysduplicate (IOFD fd);
  static void		sysopen (const char *name, int flags, int perms,
				 IOFD &newfd, unsigned &newflags);
  static bool		sysclose (IOFD fd, int *error = 0);

  unsigned		m_flags;
};

#endif // STORAGE_FACTORY_FILE_H
