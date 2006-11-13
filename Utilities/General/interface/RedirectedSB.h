#ifndef RedirectedSB_H
#define RedirectedSB_H

#include<iosfwd>
#include<sstream>
#include <streambuf>

/** rederect this streambuf to LOG
 */
template<typename LOG>
class RedirectedSB : public std::stringbuf {
public:
  typedef std::stringbuf super;

  explicit RedirectedSB(LOG * ilog=0) : m_log(ilog){}
  
  LOG * log() { return m_log;}
  LOG * log(LOG * ilog) { return m_log=ilog;}
  int sync() {
    int jj = 
      std::stringbuf::sync(); 
    if (m_log) (*m_log) << (*this).str();
    (*this).str("");
    return jj;
  }


private: 
  LOG * m_log;
};

#endif // RedirectedSB_H
