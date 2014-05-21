#ifndef Utilities_XrdAdaptor_XrdFile_h
#define Utilities_XrdAdaptor_XrdFile_h

# include "Utilities/StorageFactory/interface/Storage.h"
# include "Utilities/StorageFactory/interface/IOFlags.h"
# include "FWCore/Utilities/interface/Exception.h"
# include "XrdClient/XrdClient.hh"
# include <string>
# include <pthread.h>

class XrdFile : public Storage
{
public:
  XrdFile (void);
  XrdFile (IOFD fd);
  XrdFile (const char *name, int flags = IOFlags::OpenRead, int perms = 0666);
  XrdFile (const std::string &name, int flags = IOFlags::OpenRead, int perms = 0666);
  ~XrdFile (void);

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
  using Storage::readv;
  using Storage::write;
  using Storage::position;

  virtual bool		prefetch (const IOPosBuffer *what, IOSize n);
  virtual IOSize	read (void *into, IOSize n);
  virtual IOSize	read (void *into, IOSize n, IOOffset pos);
  virtual IOSize	readv (IOBuffer *into, IOSize n);
  virtual IOSize	readv (IOPosBuffer *into, IOSize n);
  virtual IOSize	write (const void *from, IOSize n);
  virtual IOSize	write (const void *from, IOSize n, IOOffset pos);

  virtual IOOffset	position (IOOffset offset, Relative whence = SET);
  virtual void		resize (IOOffset size);

  virtual void		close (void);
  virtual void		abort (void);

private:

  void                  addConnection(cms::Exception &);

  // "Real" implementation of readv that interacts directly with Xrootd.
  IOSize                readv_send(char **result_buffer, readahead_list &read_chunk_list, IOSize n, IOSize total_len);
  IOSize                readv_unpack(char **result_buffer, std::vector<char> &res_buf, IOSize datalen, readahead_list &read_chunk_list, IOSize n);


  XrdClient		*m_client;
  IOOffset		m_offset;
  XrdClientStatInfo	m_stat;
  bool			m_close;
  std::string		m_name;

  // We could do away with this.. if not for the race condition with LastServerResp in XrdReadv.
  pthread_mutex_t       m_readv_mutex;

};

#endif // XRD_ADAPTOR_XRD_FILE_H
