#include "func_checker.h"
[[cms::thread_guard("dummy")]] static int global_static = 23;
class Bar {
public:
Bar() {}
[[cms::thread_guard("dummy2")]] static int member_static;
void  bar()  {
	/*[[cms::thread_safe]]*/ static int local_static;
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

