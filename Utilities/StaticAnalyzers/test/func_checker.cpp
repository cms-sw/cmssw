#include "func_checker.h"
static int global_static = 23;

class Bar {
public:
Bar() {}
static int member_static;
void bar() {
	static int local_static;
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



