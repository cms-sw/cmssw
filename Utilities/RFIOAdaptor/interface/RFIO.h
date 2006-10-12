#ifndef RFIO_ADAPTOR_RFIO_H
# define RFIO_ADAPTOR_RFIO_H

//<<<<<< INCLUDES                                                       >>>>>>

/*
#include <shift.h>
# define RFIO_READOPT 1
inline int   rfio_close64(int s) { return  rfio_close(s);}
*/



#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>


//<<<<<< PUBLIC DEFINES                                                 >>>>>>
//<<<<<< PUBLIC CONSTANTS                                               >>>>>>
//<<<<<< PUBLIC TYPES                                                   >>>>>>
//<<<<<< PUBLIC VARIABLES                                               >>>>>>
//<<<<<< PUBLIC FUNCTIONS                                               >>>>>>

#ifndef  RFIO_iovec64_H
#define  RFIO_iovec64_H
extern "C"{
struct iovec64 {
        off64_t iov_base ;
        int iov_len ;
};
}
#endif

extern "C" {


    int   rfio_preseek64(int, struct iovec64 *, int);

    int   rfio_open64(const char *filepath, int flags, int mode);
    int   rfio_close(int s);
    int   rfio_close64(int s) { return  rfio_close(s);}
    int   rfio_read64(int s, void *ptr, int size);
    int   rfio_write64(int s, const void *ptr, int size);
    off64_t   rfio_lseek64(int s, off64_t  offset, int how);
    int   rfio_access(const char *filepath, int mode);
    int   rfio_unlink(const char *filepath);
    int   rfio_parse(const char *name, char **host, char **path);
    int   rfio_stat64(const char *path, struct stat64 *statbuf);
    int   rfio_fstat64(int s, struct stat64 *statbuf);
    void  rfio_perror(const char *msg);
    char *rfio_serror();
    int   rfioreadopt(int opt);
    int   rfiosetopt(int opt, int *pval, int len);
    int   rfio_mkdir(const char *path, int mode);
    void *rfio_opendir(const char *dirpath);
    int   rfio_closedir(void *dirp);
    void *rfio_readdir(void *dirp);
#   define RFIO_READOPT 1
}


// MT safe....
extern "C" {
     int  Cthread_init(void);
     int *C__rfio_errno(void);
     int *C__serrno(void);
}

#define serrno (*C__serrno())
#define rfio_errno (*C__rfio_errno())

// extern int rfio_errno;
// extern int serrno;

//<<<<<< CLASS DECLARATIONS                                             >>>>>>
//<<<<<< INLINE PUBLIC FUNCTIONS                                        >>>>>>
//<<<<<< INLINE MEMBER FUNCTIONS                                        >>>>>>

#endif // RFIO_ADAPTOR_RFIO_H
