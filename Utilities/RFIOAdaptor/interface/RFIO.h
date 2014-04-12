#ifndef RFIO_ADAPTOR_RFIO_H
# define RFIO_ADAPTOR_RFIO_H

# include <sys/stat.h>
# include <sys/uio.h>
# include <unistd.h>
# include <fcntl.h>

# define RFIO_READOPT 1
# define serrno (*C__serrno())
# define rfio_errno (*C__rfio_errno())
#ifdef __APPLE__
typedef off_t off64_t;
#endif

extern "C"
{
  // This is a RFIO-special structure like the "iovec" one
  // in sys/uio.h, but this doesn't actually exist on the system.
  struct iovec64
  {
    off64_t iov_base;
    int     iov_len;
  };

  int     rfio_preseek64(int, struct iovec64 *, int);
  int     rfio_open64(const char *filepath, int flags, int mode);
  int     rfio_close(int s);
  int     rfio_close64(int s) { return  rfio_close(s);}
  int     rfio_read64(int s, void *ptr, int size);
  int     rfio_write64(int s, const void *ptr, int size);
  off64_t rfio_lseek64(int s, off64_t  offset, int how);
  int     rfio_access(const char *filepath, int mode);
  int     rfio_unlink(const char *filepath);
  int     rfio_parse(const char *name, char **host, char **path);
  int     rfio_stat64(const char *path, struct stat *statbuf);
  int     rfio_fstat64(int s, struct stat *statbuf);
  void    rfio_perror(const char *msg);
  char *  rfio_serror();
  int     rfioreadopt(int opt);
  int     rfiosetopt(int opt, int *pval, int len);
  int     rfio_mkdir(const char *path, int mode);
  void *  rfio_opendir(const char *dirpath);
  int     rfio_closedir(void *dirp);
  void *  rfio_readdir(void *dirp);

  int     Cthread_init(void);
  int *   C__rfio_errno(void);
  int *   C__serrno(void);
}

#endif // RFIO_ADAPTOR_RFIO_H
