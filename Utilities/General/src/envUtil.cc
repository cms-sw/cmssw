#include "Utilities/General/interface/envUtil.h"
#include <cstdlib>
#include <iostream>

envUtil::envUtil(const char * envName, const char * defaultValue) :  
  envName_(envName),
  env_(::getenv(envName) ? ::getenv(envName) : "") {
  if (env_.length()==0) env_ = defaultValue;
}


envUtil::envUtil(const char * envName, const std::string & defaultValue) : 
  envName_(envName),
  env_(::getenv(envName) ? ::getenv(envName) : "") {
  if (env_.length()==0) env_ = defaultValue;
}


// avoid mess, go for a leak....
void envUtil::setEnv() {
  if (envName_.empty()) return;
  std::string * mess = new std::string(envName_);
  *mess += "=";
  *mess += env_;
  ::putenv((char*)((*mess).c_str()));
}


void envUtil::setEnv(const std::string & nval) {
  env_=nval;
  setEnv();
}

const std::string &  envUtil::getEnv(const char * envName, const char * defaultValue) {
    envName_ = envName;
    env_ = ::getenv(envName) ? ::getenv(envName) : "";
    if (env_.length()==0) env_ = defaultValue;
    return env_;
  }

envSwitch::envSwitch(const char * envName) :  envName_(envName),
					      it(::getenv(envName)!=0) {}


std::ostream & operator<<(std::ostream & co, const envUtil & eu) {
  return co << eu.getEnv();
}

std::istream & operator>>(std::istream & ci, envUtil & eu) {
  std::string m;
  ci >> m;
  eu.setEnv(m);
  return ci;
}
