#ifndef DecoratedSB_H
#define DecoratedSB_H

#include<iosfwd>
#include<sstream>
#include <streambuf>

/** a decorated sreambuf
 */
class BaseDecoratedSB : public std::stringbuf {
public:
  typedef std::stringbuf super;
  
  BaseDecoratedSB(std::streambuf * isb); 
  virtual ~BaseDecoratedSB();

  std::streambuf * sb() { return sb_;}
  std::streambuf * sb(std::streambuf * isb) { sb_ = isb; return sb_;}
  
  int sync();

private:
  virtual void pre( std::ostream & co)=0;
  virtual void post( std::ostream & co)=0;

  std::ostream me;  
  std::streambuf * sb_;
};

template<typename PRE, typename POST>
class DecoratedSB : public BaseDecoratedSB {
public:
  typedef std::stringbuf super;
  
  DecoratedSB(std::streambuf * isb, const PRE & ipre, const POST & ipost) : 
    BaseDecoratedSB(isb), pre_(ipre), post_(ipost){}
  

private:
  virtual void pre( std::ostream & co) { co<<pre_;}
  virtual void post( std::ostream & co) { co<<post_;}
  PRE pre_;
  POST post_;

};

typedef DecoratedSB<std::string, std::string>  DefDecoratedSB; 

#endif
