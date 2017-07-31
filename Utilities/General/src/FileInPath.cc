#include "Utilities/General/interface/FileInPath.h"
#include <boost/algorithm/string.hpp>
#include <iostream>
#include <fstream>


FileInPath::FileInPath(const std::string & ipath, const std::string & ifile) {
  if (ipath.empty()||ifile.empty()) return;

  std::vector<String> directories;
  boost::split(directories, ipath, [](char iC) { return iC == ':';});
  
  for (auto const& d :directories) {
    m_file = d; m_file += "/"; m_file += ifile;
    m_in = std::make_unique<std::ifstream>(m_file.c_str());
    if (m_in->good()) break;
  }
  if (!m_in->good()) { m_in.release(); m_file="";}
}

FileInPath::FileInPath(const FileInPath& rh ) : 
  m_file(rh.m_file)
{
  if (rh.m_in.get()) {
    m_in = std::make_unique<std::ifstream>(m_file.c_str());
  }
}
  

FileInPath & FileInPath::operator=(const FileInPath& rh ) {
  m_file = rh.m_file;
  if (rh.m_in.get()&&(!m_file.empty())) {
    m_in = std::make_unique<std::ifstream>(m_file.c_str());
  }
  return *this;
}

FileInPath::~FileInPath() {}

