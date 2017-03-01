#ifndef STORAGE_FACTORY_LOCAL_FILE_SYSTEM_H
# define STORAGE_FACTORY_LOCAL_FILE_SYSTEM_H
# include <vector>
# include <string>
#include <utility>

struct stat;
struct statfs;
struct mntent;

class LocalFileSystem
{
  struct FSInfo;
public:
  LocalFileSystem(void);
  ~LocalFileSystem(void);

  bool		isLocalPath(const std::string &path) const;
  std::pair<std::string, std::string>	findCachePath(const std::vector<std::string> &paths, double minFreeSpace) const;

private:
  int		readFSTypes(void);
  FSInfo *	initFSInfo(void *p);
  int		initFSList(void);
  int		statFSInfo(FSInfo *i) const;
  FSInfo *	findMount(const char *path, struct statfs *sfs, struct stat *s, std::vector<std::string> &) const;

  std::vector<FSInfo *> fs_;
  std::vector<std::string> fstypes_;

  // undefined, no semantics
  LocalFileSystem(LocalFileSystem &);
  void operator=(LocalFileSystem &);
};

#endif // STORAGE_FACTORY_LOCAL_FILE_SYSTEM_H
