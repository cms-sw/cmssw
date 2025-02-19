#include "Utilities/General/interface/Tokenizer.h"
#include "Utilities/General/interface/FileInPath.h"
#include <iostream>
#include <fstream>

const std::string FileInPath::semicolon(":");

FileInPath::FileInPath(const std::string & ipath, const std::string & ifile) {
  init(ipath, ifile);
}

void FileInPath::init(const std::string & ipath, const std::string & ifile) {
  if (ipath.empty()||ifile.empty()) return;

  directories = Tokenizer(semicolon,ipath);
  typedef std::vector<std::string>::const_iterator Itr;
  for (Itr d=directories.begin(); d!=directories.end(); d++) {
    file = *d; file += "/"; file += ifile;
    in = own_ptr<std::ifstream>(new std::ifstream(file.c_str()));
    if (in->good()) break;
  }
  if (!in->good()) { in.release(); file="";}
}

FileInPath::FileInPath(const FileInPath& rh ) : 
  directories(rh.directories), file(rh.file) {
  if (rh.in.get()) in = own_ptr<std::ifstream>(new std::ifstream(file.c_str()));
}
  

FileInPath & FileInPath::operator=(const FileInPath& rh ) {
  directories = rh.directories; 
  file = rh.file;
  if (rh.in.get()&&(!file.empty())) in = own_ptr<std::ifstream>(new std::ifstream(file.c_str()));
  return *this;
}

FileInPath::~FileInPath() {}

