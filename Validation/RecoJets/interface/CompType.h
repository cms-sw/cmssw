#ifndef CompType_h
#define CompType_h

#include <memory>
#include <string>
#include <vector>

class CompType {
  
 public:
  enum {kRatio, kRelDiff};
  
 public:
  CompType(int t):type_(t){};
  ~CompType(){};
  double operator()(const double&, const double& );
  
 private:
  unsigned int type_;
};

inline double 
CompType::operator()(const double& ref, const double& sample)
{
  double value=0;
  switch( type_ ){
  case CompType::kRatio:
    value=sample/ref;
    break;

  case CompType::kRelDiff:
    value=(sample-ref)/ref;
    break;

  default:
    break;
  }  
  return value;
}

#endif
