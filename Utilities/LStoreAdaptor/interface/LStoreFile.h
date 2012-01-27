#ifndef LSTORE_ADAPTOR_LSTORE_FILE_H
# define LSTORE_ADAPTOR_LSTORE_FILE_H

# include "Utilities/StorageFactory/interface/Storage.h"
# include "Utilities/StorageFactory/interface/IOFlags.h"
# include <string>
#include <pthread.h>
class LStoreFile : public Storage
{
public:
  LStoreFile (void);
  LStoreFile (void * fd);
  LStoreFile (const char *name, int flags = IOFlags::OpenRead, int perms = 0666);
  LStoreFile (const std::string &name, int flags = IOFlags::OpenRead, int perms = 0666);
  ~LStoreFile (void);

  virtual void	create (const char *name,
    			bool exclusive = false,
    			int perms = 0666);
  virtual void	create (const std::string &name,
    			bool exclusive = false,
    			int perms = 0666);
  virtual void	open (const char *name,
    		      int flags = IOFlags::OpenRead,
    		      int perms = 0666);
  virtual void	open (const std::string &name,
    		      int flags = IOFlags::OpenRead,
    		      int perms = 0666);

  using Storage::read;
  using Storage::write;
  using Storage::position;

  virtual IOSize	read (void *into, IOSize n);
  virtual IOSize	write (const void *from, IOSize n);

  virtual IOOffset	position (IOOffset offset, Relative whence = SET);
  virtual void		resize (IOOffset size);

  virtual void		close (void);
  virtual void		abort (void);
  
  class MutexWrapper {
	public:
	  MutexWrapper( pthread_mutex_t * lock );
	  ~MutexWrapper();
	  pthread_mutex_t * m_lock;
  };

  static pthread_mutex_t m_dlopen_lock;

private:
  void *		m_fd;
  bool			m_close;
  std::string		m_name;
  void loadLibrary();
  void closeLibrary();
  void * m_library_handle;
  bool m_is_loaded;

  // Prototypes for 

  int32_t (*redd_init)();
  int64_t (*redd_read)(void *, char*, int64_t); 
  int32_t (*redd_close)(void *);
  int64_t (*redd_lseek)(void *, int64_t, uint32_t);
  void * (*redd_open)(const char *, int32_t, int32_t );
  int64_t (*redd_write)(void *, const char *, int64_t);
  int32_t (*redd_term)();
  int32_t (*redd_errno)();
  const std::string & (*redd_strerror)();
};

#endif // LSTORE_ADAPTOR_LSTORE_FILE_H

