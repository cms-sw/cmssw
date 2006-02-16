#ifndef _DetLayers_ENUMERATORS_H_ 
#define _DetLayers_ENUMERATORS_H_

/// This file will be moved to a different location soon.

using namespace std;
#include <iosfwd>
#include <ostream>

/** Global enumerators for Det types.
 */
enum Part { barrel, forward};
enum Module {pixel=1, silicon=2, msgc=3, dt=4, csc=5, rpc=6};

/** overload << for correct output of the enumerators 
 *  (e.g. to get "barrel" instead of "0")
 */
ostream& operator<<( ostream& s, Part p);
ostream& operator<<( ostream& s, Module m);


#endif
