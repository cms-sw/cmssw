#ifndef UTILITIES_GENERAL_FILEINPATH_H
#define UTILITIES_GENERAL_FILEINPATH_H
//
//
//   V 0.0 
//
#include<string>
#include<vector>
#include<iosfwd>
#include <memory>

/** open the first file found in a ":"-separated list of files (path)
 */
class FileInPath {
public:
  typedef std::string String;
public:
  /// constructor
  FileInPath(const String & ipath, const String & ifile);

  // deep copy constructor
  FileInPath(const FileInPath& rh );
  
  /// and usual operator 
  FileInPath & operator=(const FileInPath& rh );

  /// destructor
  ~FileInPath();

  /// return stream
  std::ifstream * operator()() { return m_in.get();}
  
  /// return full name
  const String & name() const { return m_file;}

private:
  String m_file;
  std::unique_ptr<std::ifstream> m_in;

};

#endif // UTILITIES_GENERAL_FILEINPATH_H
