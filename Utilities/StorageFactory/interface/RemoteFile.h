#ifndef STORAGE_FACTORY_REMOTE_FILE_H
#define STORAGE_FACTORY_REMOTE_FILE_H

#include "Utilities/StorageFactory/interface/File.h"
#include <string>
#include <memory>

class RemoteFile : protected File {
public:
  ~RemoteFile(void) override { remove(); }

  static int local(const std::string &tmpdir, std::string &temp);
  static std::unique_ptr<Storage> get(int localfd, const std::string &name, char **cmd, int mode);

protected:
  void close(void) override;
  void abort(void) override;

private:
  RemoteFile(IOFD fd, const std::string &name);
  void remove(void);
  std::string name_;
};

#endif  // STORAGE_FACTORY_REMOTE_FILE_H
