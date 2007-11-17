#ifndef STORAGE_FACTORY_SYS_FILE_H
# define STORAGE_FACTORY_SYS_FILE_H

# include <unistd.h>
# include <sys/stat.h>
# include <fcntl.h>
# include <utime.h>
# include <limits.h>
# include <cerrno>
# include <cstdlib>

# if !defined O_SYNC && defined O_SYNCIO
#  define O_SYNC O_SYNCIO
# endif

# if !defined O_NONBLOCK && defined O_NDELAY
#  define O_NONBLOCK O_NDELAY
# endif

# ifndef O_NONBLOCK
#  define O_NONBLOCK 0
# endif

#endif // STORAGE_FACTORY_SYS_FILE_H
