#ifndef RFIO_ADAPTOR_RFIO_H
# define RFIO_ADAPTOR_RFIO_H

//<<<<<< INCLUDES                                                       >>>>>>

#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>

//<<<<<< PUBLIC DEFINES                                                 >>>>>>
//<<<<<< PUBLIC CONSTANTS                                               >>>>>>
//<<<<<< PUBLIC TYPES                                                   >>>>>>
//<<<<<< PUBLIC VARIABLES                                               >>>>>>
//<<<<<< PUBLIC FUNCTIONS                                               >>>>>>

extern "C" {
    int   rfio_open(const char *filepath, int flags, int mode);
    int   rfio_close(int s);
    int   rfio_read(int s, void *ptr, int size);
    int   rfio_write(int s, const void *ptr, int size);
    int   rfio_lseek(int s, int offset, int how);
    int   rfio_access(const char *filepath, int mode);
    int   rfio_unlink(const char *filepath);
    int   rfio_parse(const char *name, char **host, char **path);
    int   rfio_stat(const char *path, struct stat *statbuf);
    int   rfio_fstat(int s, struct stat *statbuf);
    void  rfio_perror(const char *msg);
    char *rfio_serror();
    int   rfiosetopt(int opt, int *pval, int len);
    int   rfio_mkdir(const char *path, int mode);
    void *rfio_opendir(const char *dirpath);
    int   rfio_closedir(void *dirp);
    void *rfio_readdir(void *dirp);
#   define RFIO_READOPT 1
}

extern int rfio_errno;
extern int serrno;

//<<<<<< CLASS DECLARATIONS                                             >>>>>>
//<<<<<< INLINE PUBLIC FUNCTIONS                                        >>>>>>
//<<<<<< INLINE MEMBER FUNCTIONS                                        >>>>>>

#endif // RFIO_ADAPTOR_RFIO_H
