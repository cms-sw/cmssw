#include "Utilities/General/interface/precomputed_value_sort.h"
#include "Utilities/General/interface/GeometricSorting.h"
#include "DataFormats/GeometrySurface/interface/GloballyPositioned.h"
#include <iostream>
#include <iterator>

using namespace std;
using namespace geomsort;


typedef GloballyPositioned<float> Gp; // the object to be sorted
typedef Gp::PositionType PositionType;
typedef Gp::RotationType RotationType;

// A simple helper for printing the object
struct dump {
  void operator() (const Gp& o) {
    cout << o.position()
	 << " R  : " << o.position().perp()
	 << " Phi: " << o.position().phi()
	 << " Z  : " << o.position().z()
	 << endl;
  }
  void operator() (const Gp* o) {
    operator()(*o);
  }  
};


int main() {

  //
  // Example of sorting a vector of objects store by value.
  //

  // Fill the vector to be sorted
  vector<Gp> v;
  v.push_back(Gp(PositionType(2,1,1),RotationType()));
  v.push_back(Gp(PositionType(1,1,2),RotationType()));
  v.push_back(Gp(PositionType(1,2,3),RotationType()));
  v.push_back(Gp(PositionType(2,2,4),RotationType()));


  cout << "Original  vector: " << endl;
  for_each(v.begin(),v.end(), dump());

  cout << "Sort in R       : " << endl;
  // Here we sort in R
  precomputed_value_sort(v.begin(),v.end(),ExtractR<Gp>());
  for_each(v.begin(),v.end(), dump());

  cout << "Sort in phi     : " << endl;
  // Here we sort in phi
  precomputed_value_sort(v.begin(),v.end(),ExtractPhi<Gp>());
  for_each(v.begin(),v.end(), dump());

  cout << "Sort in z       : " << endl;
  // Here we sort in Z
  precomputed_value_sort(v.begin(),v.end(),ExtractZ<Gp>());
  for_each(v.begin(),v.end(), dump());



  //
  // Now do the same with a vector of pointers
  //
  cout << endl << "Again with pointers" << endl;

  vector<const Gp*> vp;
  for (vector<Gp>::const_iterator i=v.begin(); i!=v.end(); i++){
    vp.push_back(&(*i));
  }

  cout << "Sort in R       : " << endl;
  // Here we sort in R
  precomputed_value_sort(vp.begin(),vp.end(),ExtractR<Gp>());
  for_each(vp.begin(),vp.end(), dump());

  cout << "Sort in phi     : " << endl;
  // Here we sort in phi
  precomputed_value_sort(vp.begin(),vp.end(),ExtractPhi<Gp>());
  for_each(vp.begin(),vp.end(), dump());

  cout << "Sort in z       : " << endl;
  // Here we sort in Z
  precomputed_value_sort(vp.begin(),vp.end(),ExtractZ<Gp>());
  for_each(vp.begin(),vp.end(), dump());


  return 0;  
}



