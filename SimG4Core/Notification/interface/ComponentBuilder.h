#ifndef SimG4Core_ComponentBuilder_H
#define SimG4Core_ComponentBuilder_H

#include <string>

template <class C,class T> class ComponentBuilder
{
public:
    virtual C * constructComponent(T) = 0;
    virtual std::string myName() = 0;
};

#endif

