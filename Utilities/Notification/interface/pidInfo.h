#ifndef pidInfo_h
#define pidInfo_h


#include <string>
#include <iosfwd>


/** class to print out process size and resident size
  clearly not elegent, but it works!
  you supply a string so you can see in the output where you called it from
*/
class pidInfo
{
public:
  ///
  pidInfo();

  explicit pidInfo(const std::string & s);
  explicit pidInfo(std::ostream & co, const std::string & s="");

  ///
  ~pidInfo(){}

  unsigned int totalMemory() const { return pagesz*memsz;}
  unsigned int residentMemory() const { return pagesz*rsssz;}

private:


  void load(std::ostream * co);

  // this is wrong as it is os specific not compiler specific...
#ifdef __SUNPRO_CC
  string command(const std::string & pid);
#endif

  unsigned int memsz;
  unsigned int rsssz;

  static const unsigned int pagesz;


}; 

#endif






