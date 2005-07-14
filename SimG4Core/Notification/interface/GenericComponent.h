#ifndef Frappe_GenericComponent_h 
#define Frappe_GenericComponent_h 

#include "SealKernel/Component.h"
#include "SealKernel/Context.h"
#include "SealKernel/ComponentLoader.h"
# include "PluginManager/PluginFactory.h"

#include <typeinfo>
#include<memory>

#include<iostream>

namespace frappe {



  template <class S>
  inline void queryInChildren (const seal::Context & c, std::vector< seal::IHandle<S> > &matches)
  {
    c.queryLocal(matches);
    for (size_t i = 0; i < c.children(); ++i) {
      queryInChildren(*c.child(i),matches); 
    }
  }

  struct ComponentNotCreated {};
  struct NoComponent{};
  
  // workaround to aovid assert on MAC....
  template<typename T> struct Hello {
    Hello(const char* c) :
      ck(seal::ContextKey::find(c)) {
      if (ck.value()==seal::ContextKey::KEY_NONE) {
	static seal::ContextKeyDef local(c);
	ck = local;
      }
      std::cout << "Hallo " << c << std::endl;
    }
    seal::ContextKey ck;
  };


  
  class ComponentFactory : public seal::PluginFactory< seal::Component* ( seal::Context&, const std::string&) > {
  public:
    static ComponentFactory * get (void) {
      static ComponentFactory me;
      return &me;
    }
  private: 
    ComponentFactory() : 
      seal::PluginFactory< seal::Component* ( seal::Context&, const std::string&) >("frappe Component Factory"){
    }
  };

  template<typename B>
  class ServiceFactory : public seal::PluginFactory< B* ( seal::Context*) > {
  public:
    static ServiceFactory<B> * get (void) {
      std::cout << "get for factory for " <<  typeid(B).name() << std::endl;
      static ServiceFactory<B> me;
      return &me;
    }
  private: 
    ServiceFactory() : seal::PluginFactory< B* ( seal::Context*) >(std::string("frappe Component") + std::string(typeid(B).name())){
      std::cout << "factory instantiated for " <<  typeid(B).name() << std::endl;
    }
  };

  // just for the sake of forward-declared B
  template<typename B>
  struct Key {
    static const char* classContextLabel(void) {
      return typeid(Key<B>).name();
    }   
    static const char* name(void) {
      return typeid(Key<B>).name();
    }   

  };

  template<typename B>
  class LazyComponent : public seal::Component {
  public:                                     
    typedef LazyComponent<B> self;
    
    B & operator()(){ return get();}

    B & operator*(){ return get();}

    B & get() { 
      check();
      return *t;
    }

    // def constr...
    LazyComponent() {}

    // lazy constructor
    LazyComponent(seal::Context& icontext, const std::string & iname) :
      seal::Component(&icontext, classContextKey()), name(iname){
      std::cout << "creating a LazyComponent for " << name 
		<< " in " << context() << std::endl;
    }

    // immediate constructor
    LazyComponent(seal::Context& context, B *b) :
      seal::Component(&context, classContextKey()), t(b){}
    
    static const char* classContextLabel(void) {
      return typeid(self).name();
    }   
    static seal::ContextKey classContextKey(void) {
      static Hello<self > a(classContextLabel());
      //static seal::ContextKeyDef local(classContextLabel());
      //return local;
      return a.ck;
    }

    static bool exists(seal::Context& context) {
      std::vector< seal::IHandle<self> > vh;
      context.queryLocal (vh);
      return !vh.empty();
    }

    

  private:
    void check() {
      if (!t.get()) {
	std::cout << "creating a Component for " << name 
		  << " in " << context() << std::endl;
	t.reset((*ServiceFactory<B>::get()).create(name,context()));
      }
      if (!t.get()) throw ComponentNotCreated();
    }

  private:
    std::string name;
    std::auto_ptr<B>  t;
  };




  class Client {
  public:
    Client() : m_context(0){}
    explicit Client(seal::Context& icontext) :
      m_context(&icontext){}
    
    virtual ~Client(){}
    
    seal::Context& context() const { return *m_context;}
    
    //get service of type C
    template<typename C> C & component() const {
      seal::Handle<LazyComponent<C> > lc = context().template component<LazyComponent<C> >();
      if (!lc) throw NoComponent(); // <C>(m_context);
      return (*lc).get();
    }
    
    template<typename C> bool exists() const {
      return LazyComponent<C>::exists(context());
    }

  private:
    
    seal::Context* m_context;
    
  };
  
  class Configurator : public Client {
  public:


    static void load( const std::string & baseName, 
		      seal::Context & icontext, const std::string & sname) {
      (*frappe::ComponentFactory::get()).create(baseName,icontext, sname);
    }

    Configurator() {}
    explicit Configurator(seal::Context& icontext) :
      Client(icontext){}
    
    // add instance of Service of type C
    template<typename C> bool addComponent(C* c)  {
      if (exists<C>()) return false; // should delete c or replace?
      return new LazyComponent<C>(context(),c);
    }

    /* add instance of Service of type given by typeinfo
       using reflection
    */
    bool addComponent(const std::type_info & ti, void * v);

    
    // configure context to load service of type C with name iname
    template<typename C> bool configure(const std::string & iname)  {
      if (exists<C>()) return false;
      return new LazyComponent<C>(context(),iname);      
    }

    // configure context to load service of Mnemonic-Type baseName with name sName
    bool configure( const std::string & baseName, const std::string & sName) {
      return (*frappe::ComponentFactory::get()).create(baseName,context(), sName);
    }


  };
  
  // some trivial yet useful services

  struct ContextName : public std::string {
    ContextName() {}

    ContextName(const char * iname) : std::string(iname){}
    ContextName(const std::string & iname) : std::string(iname){}

  };



  // old stuff --------------------------------------------------------

  template<typename S> struct GetComponent { 
    seal::Handle<S> operator()(const seal::Context & icontext) const {
      seal::Handle<seal::Component> h = icontext.component(S::classContextKey());
      return dynamic_cast<S*>(h.get());
    }
  };

  template<typename B>
  class RefComponent : public seal::Component {
  public:

    typedef RefComponent<B> self;

    static B & get(seal::Context & icontext) {
      // static GetComponent<self> c;
      return (*icontext.template component<self>())();
    }

    static const char* classContextLabel(void) {
      return typeid(self).name();
    }   
    static seal::ContextKey classContextKey(void) {
      static Hello<self > a(classContextLabel());
      //static seal::ContextKeyDef local(classContextLabel());
      //return local;
      return a.ck;
    }

    RefComponent(B & it,  seal::Context & icontext) : 
      seal::Component(&icontext, classContextKey()), t(&it) {}
    B & operator()(){ return *t;}
  private:
    B * t; 
  };  
  

 

  template<typename T, typename B=T>
  class GenericComponent : public seal::Component {
  public:                                     
    typedef GenericComponent<T,B> self;

    static B & loadUnique(seal::Context & icontext) {
      B & b = load(icontext);
      new RefComponent<B>(b,icontext);
      return b;
    }
    static B & load(seal::Context & icontext) {
      icontext.template component<seal::ComponentLoader>()->load(&icontext,self::classContextLabel());
      return self::get(icontext);
    }

    static B & get(seal::Context & icontext) {
      static GetComponent<self> c;
      return (*c(icontext))();
    }

    static const char* classContextLabel(void) {
      return typeid(self).name();
    }   
    static seal::ContextKey classContextKey(void) {
      static Hello<self > a(classContextLabel());
      //static seal::ContextKeyDef local(classContextLabel());
      //return local;
      return a.ck;
    }
   
    GenericComponent(seal::Context* context) :
      seal::Component(context, classContextKey()), t(new T(context)) {}
    
    GenericComponent(seal::Context* context, const std::string & key)
      : seal::Component(context, key) , t(new T(context)) {}
    
    GenericComponent(B * it,  seal::Context* context) :
      seal::Component(context, classContextKey()), t(it) {}

    virtual ~GenericComponent(){
      std::cout << "delete  " << typeid(*t).name() << " at " << t.get() << std::endl;
    }

    B & operator()(){ return *t;}

  private:
    std::auto_ptr<B>  t;
  };




}



#endif
