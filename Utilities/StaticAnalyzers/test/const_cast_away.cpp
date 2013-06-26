
#include <string>


int main()
{
    std::string s = "23";
    std::string const& r_const = s;
	std::string const* p_const = &s;
    
    // will produce a warning only by ConstCastAwayChecker
    std::string & r = (std::string &) ( r_const );
	std::string * p = (std::string *) ( p_const );

	// must not produce a warning
	std::string const& r_const_again = ( std::string const&) ( r_const );

    return 0;
}
