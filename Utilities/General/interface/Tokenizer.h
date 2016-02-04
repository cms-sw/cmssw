#ifndef UTILITIES_GENERAL_TOKENIZER_H
#define UTILITIES_GENERAL_TOKENIZER_H

#include <string>
#include <vector>

/** Tokenize "input" in a vector<string> at each occurence of "sep"
 */
class Tokenizer : public std::vector<std::string> {
public:
  typedef std::vector<std::string> super;
  Tokenizer(const std::string & sep, const std::string & input, bool alsoempty=true);
  
  void join(std::string & out, const std::string & sep, bool alsoempty=true) const;

private:

};

#endif // UTILITIES_GENERAL_TOKENIZER_H
