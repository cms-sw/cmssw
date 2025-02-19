#ifndef stringTools_h
#define stringTools_h

#include<string>

/** replace "gone" with "it" in "input"
if multiple=true reconsider the string after each replacement
 */
int replace(std::string & input, const std::string& gone, const std::string& it, bool multiple=false);

/** replace everything between "first" and "last" with "it" in "input" 
if multiple=true reconsider the string after each replacement
 */
int replaceRange(std::string & input, const std::string&first, const std::string& last, const std::string& it, bool multiple=false);


/*
  remove leading and trailing "blanks"
*/
void strip(std::string & input, const std::string& blanks=" \n\t");


/** return the new substring...
 */
class stringUpdate{
public:
  stringUpdate(const std::string & is) : s(is), old(0){}

  std::string operator()() { 
    if (s.size()>old) {
      size_t d = old;
      old = s.size();
      return s.substr(d);
    }
    else return std::string();
  }

  void reset() { old=0;}
  bool updated() const { return s.size()>old;}

private: 

  const std::string & s;
  size_t old;
};


#endif // stringTools_h
