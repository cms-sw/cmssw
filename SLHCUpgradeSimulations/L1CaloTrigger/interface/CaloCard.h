/* SLHC Calo Trigger 
Template class for all lattice operations
Defines the general card ..

M.Bachtis,S.Dasu. University of Wisconsin-Madison
*/


#ifndef CaloCard_h
#define CaloCard_h

#include <vector>
#include <map>
#include <string>

template <class T>
class CaloCard
{

 protected:
  //Define the Map of the Lattice
  std::map<int,T> lattice_;


 public:
  CaloCard()
    {

    }



  void reset()//Clear the Lattice
    {
      lattice_.clear();
    }


};

#endif




