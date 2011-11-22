#include "Utilities/StorageFactory/interface/RemoteFile.h"
#include "Utilities/StorageFactory/src/Throw.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <sys/wait.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <cerrno>
#include <cassert>
#include <spawn.h>
#include <unistd.h>
#include <ostream>
#include <cstring>
#if __APPLE__
# include <crt_externs.h>
# define environ (*_NSGetEnviron())
#endif

static std::string
join (char **cmd)
{
  size_t size = 0;
  for (char **p = cmd; p && p[0]; ++p)
    size += 1 + strlen(*p);

  std::string result;
  result.reserve (size);

  for (char **p = cmd; p && p[0]; ++p)
  {
    if (p != cmd)
      result += ' ';
    result += *p;
  }

  return result;
}

RemoteFile::RemoteFile (IOFD fd, const std::string &name)
  : File (fd),
    name_ (name)
{}

void
RemoteFile::remove (void)
{ unlink (name_.c_str()); }

void
RemoteFile::close (void)
{ remove(); File::close (); }

void
RemoteFile::abort (void)
{ remove(); File::abort (); }

int
RemoteFile::local (const std::string &tmpdir, std::string &temp)
{
  // Download temporary files to the current directory by default.
  // This is better for grid jobs as the current directory is
  // likely to have more space, and is more optimised for
  // large files, and is cleaned up after the job.
  if (tmpdir.empty () || tmpdir == ".")
  {
    size_t len = pathconf (".", _PC_PATH_MAX);
    char   *buf = (char *) malloc (len);
    getcwd (buf, len);

    temp.reserve (len + 32);
    temp = buf;
    free (buf);
  }
  else
  {
    temp.reserve (tmpdir.size() + 32);
    temp = tmpdir;
  }
  if (temp[temp.size()-1] != '/')
    temp += '/';

  temp += "storage-factory-local-XXXXXX";
  temp.c_str(); // null terminate for mkstemp

  umask(0x022);
  int fd = mkstemp (&temp[0]);
  if (fd == -1)
    throwStorageError("RemoteFile", "Calling RemoteFile::local()", "mkstemp()", errno);

  return fd;
}

Storage *
RemoteFile::get (int localfd, const std::string &name, char **cmd, int mode)
{
  // FIXME: On write, create a temporary local file open for write;
  // on close, trigger transfer to destination.  If opening existing
  // file for write, may need to first download.
  assert (! (mode & (IOFlags::OpenWrite | IOFlags::OpenCreate)));

  pid_t	 pid = -1;
  int    rc = posix_spawnp (&pid, cmd[0], 0, 0, cmd, environ);

  if (rc == -1)
  {
    int errsave = errno;
    ::close (localfd);
    unlink (name.c_str());
    throwStorageError ("RemoteFile", "Calling RemoteFile::get()", "posix_spawnp()", errsave);
  }

  pid_t rcpid;
  do
    rcpid = waitpid(pid, &rc, 0);
  while (rcpid == (pid_t) -1 && errno == EINTR);

  if (rcpid == (pid_t) -1)
  {
    int errsave = errno;
    ::close (localfd);
    unlink (name.c_str());
    throwStorageError ("RemoteFile", "Calling RemoteFile::get()", "waitpid()", errsave);
  }

  if (WIFEXITED(rc) && WEXITSTATUS(rc) == 0)
    return new RemoteFile (localfd, name);
  else
  {
    ::close (localfd);
    unlink (name.c_str());
    cms::Exception ex("RemoteFile");
    ex << "'" << join(cmd) << "'"
       << (WIFEXITED(rc) ? " exited with exit code "
	   : WIFSIGNALED(rc) ? " died from signal "
	   : " died for an obscure unknown reason with exit status ")
       << (WIFEXITED(rc) ? WEXITSTATUS(rc)
	   : WIFSIGNALED(rc) ? WTERMSIG(rc)
	   : rc);
    ex.addContext("Calling RemoteFile::get()");
    throw ex;
  }
}
