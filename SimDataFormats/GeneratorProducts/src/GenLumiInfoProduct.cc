#include <iostream>
#include <algorithm> 

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "SimDataFormats/GeneratorProducts/interface/GenLumiInfoProduct.h"

using namespace edm;
using namespace std;


const bool operator < (const GenLumiInfoProduct::ProcessInfo& lhs, const GenLumiInfoProduct::ProcessInfo& rhs) 
{ return (lhs.process() < rhs.process()); }

const bool operator != (const GenLumiInfoProduct::ProcessInfo& lhs, const GenLumiInfoProduct::ProcessInfo& rhs) 
{ bool condition = (lhs.process() != rhs.process()) ||
    (lhs.lheXSec() != rhs.lheXSec());
  return condition; 
}

const bool operator == (const GenLumiInfoProduct::ProcessInfo& lhs, const GenLumiInfoProduct::ProcessInfo& rhs)
{ 
  bool condition=
    (lhs.process() == rhs.process() && lhs.lheXSec() == rhs.lheXSec()
     && lhs.tried() == rhs.tried() && lhs.selected() == rhs.selected()
     && lhs.killed() == rhs.killed() && lhs.accepted() == rhs.accepted() 
     && lhs.acceptedBr() == rhs.acceptedBr()); 
  return condition;
}

const bool operator !=(const GenLumiInfoProduct& lhs, const GenLumiInfoProduct& rhs)
{
  std::vector<GenLumiInfoProduct::ProcessInfo> lhsVector = lhs.getProcessInfos();
  std::vector<GenLumiInfoProduct::ProcessInfo> rhsVector = rhs.getProcessInfos();
  std::sort(lhsVector.begin(),lhsVector.end());
  std::sort(rhsVector.begin(),rhsVector.end());
  unsigned int lhssize=lhsVector.size();
  unsigned int rhssize=rhsVector.size();  
  bool condition= (lhs.getHEPIDWTUP() != rhs.getHEPIDWTUP()) ||
    (lhssize != rhssize);
  bool fail=false;
  if(!condition)
    {
      for(unsigned int i=0; i<lhssize; i++){
	if(lhsVector[i] != rhsVector[i])
	  {
	    fail=true;
	    break;
	  }

      }

    }
  return (condition || fail);
  
}


const bool operator ==(const GenLumiInfoProduct& lhs, const GenLumiInfoProduct& rhs)
{
  std::vector<GenLumiInfoProduct::ProcessInfo> lhsVector = lhs.getProcessInfos();
  std::vector<GenLumiInfoProduct::ProcessInfo> rhsVector = rhs.getProcessInfos();
  std::sort(lhsVector.begin(),lhsVector.end());
  std::sort(rhsVector.begin(),rhsVector.end());
  unsigned int lhssize=lhsVector.size();
  unsigned int rhssize=rhsVector.size();  

  bool condition= (lhs.getHEPIDWTUP() == rhs.getHEPIDWTUP()) &&
    (lhssize == rhssize);
  unsigned int passCounts=-999;
  if(condition)
    {
      for(unsigned int i=0; i<lhssize; i++){
	if(lhsVector[i] == rhsVector[i])
	  passCounts++;
      }
    }
  return (condition && (passCounts==lhssize));
  
}


GenLumiInfoProduct::GenLumiInfoProduct() :
  hepidwtup_(-1)
{
  internalProcesses_.clear();
  
}

GenLumiInfoProduct::GenLumiInfoProduct(const int id) :
  hepidwtup_(id)
{
  internalProcesses_.clear();
}

GenLumiInfoProduct::GenLumiInfoProduct(GenLumiInfoProduct const &other) :
  hepidwtup_(other.hepidwtup_),
  internalXSec_(other.internalXSec_),        
  internalProcesses_(other.internalProcesses_)
{
}

GenLumiInfoProduct::~GenLumiInfoProduct()
{
}

bool GenLumiInfoProduct::mergeProduct(GenLumiInfoProduct const &other)
{
  if ( (*this) != other)
    {

      edm::LogWarning("GenLumiInfoProduct|ProductsNotMergeable")
	<< "You are merging runs with different cross-sections"
	"The resulting cross-section will not be consistent." << std::endl;

      return false;
    }
  return true;
}

bool GenLumiInfoProduct::isProductEqual(GenLumiInfoProduct const &other) const
{
  return ((*this) == other);
}

