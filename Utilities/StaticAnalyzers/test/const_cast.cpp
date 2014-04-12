
#include <string>


int main()
{
    std::string s = "23";
    std::string const& r_const = s;
    
    // will produce a warning by ConstCastChecker
    // and ConstCastAwayChecker
    std::string & r = const_cast< std::string & >( r_const );

    return 0;
}
