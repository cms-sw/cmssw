#ifndef Utilities_CMSexception_H
#define Utilities_CMSexception_H
//
//  VI 1.0  3/4/2002
//  exception with trace back and chaining
//

# include "FWCore/Utilities/interface/Exception.h"



/** base CMSexception
 */
class CMSexception : public cms::Exception {
public:
  CMSexception() throw() : cms::Exception("") {}
  virtual ~CMSexception() throw() {}
  virtual const char* what() const throw() = 0;
  virtual std::string explainSelf (void) const { return what();}
};

/** fast exception
 */
class Fastexception: public CMSexception  {
public:
  Fastexception() throw() {}
  virtual ~Fastexception() throw() {}
  virtual const char* what() const throw() { return 0;}
  virtual std::string explainSelf (void) const { return "";}
  virtual cms::Exception *     clone (void) const { return new Fastexception(*this);}
  virtual void        rethrow (void) { throw *this;}
};


#include <string>

/** base generic exception
 */
class BaseGenexception: public CMSexception  {
public:
  BaseGenexception() throw();
  BaseGenexception(const std::string & mess) throw();
  virtual ~BaseGenexception() throw();
  virtual const char* what() const throw() { return message.c_str();}
  virtual cms::Exception *     clone (void) const { return new BaseGenexception(*this);}
  virtual void        rethrow (void) { throw *this;}

private:
  std::string message;
};



#include "Utilities/General/interface/own_ptr.h"
#include<iosfwd>

/** cms generic  exception
 */
class Genexception: public BaseGenexception  {
public:
  Genexception() throw();
  Genexception(const std::string & mess) throw();
  virtual ~Genexception() throw();
  const std::string & trace() const throw() { return trace_;}

  void add(Genexception * in) throw();

  void dump(std::ostream & o, bool it=false) const throw(); 
  virtual cms::Exception *  clone (void) const { return new Genexception(*this);}
  virtual void        rethrow (void) { throw *this;}

private:
  void traceit() throw();

private:
  std::string trace_;

  own_ptr<Genexception> next;
};

namespace Capri {

  class Error : public Genexception {
  public:
    Error(const std::string & level, const std::string & mess) throw();
  };
  
  
  class Warning : public Error {
  public: 
    explicit Warning(const std::string & mess) throw() : Error("Warning",mess){}
  };
  
  class Severe : public Error {
  public:
    explicit Severe(const std::string & mess) throw() : Error("Severe",mess){}
  };
  
  class Fatal : public Error {
  public:
    explicit Fatal(const std::string & mess) throw() : Error("Fatal",mess){}
  };


}

/** terminate processing
 */
class GenTerminate: public Genexception  {
public:
  GenTerminate() throw();
  GenTerminate(const std::string & mess) throw();
  virtual ~GenTerminate() throw();

};


/** not a real exception...
 */
class Success {
public:
  Success();
  ~Success();

private:
  std::string message;
};

#endif

