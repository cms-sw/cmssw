#include "FWCore/Utilities/interface/thread_safety_macros.h"
#include "func_checker.h"

CMS_THREAD_GUARD(dummy) static int global_static = 23;
class Bar {
public:
Bar() {}
CMS_THREAD_GUARD(dummy2) static int member_static;
void  bar()  {
	/*CMS_THREAD_SAFE*/ static int local_static;
	member_static = global_static;
	local_static = global_static; 
	member_static = external_int;
	local_static = external_int;
}

};

int main()
{
return 0;
}

