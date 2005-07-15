#ifndef Utilities_DispatcherObserver_H
#define Utilities_DispatcherObserver_H

#include "Utilities/Notification/interface/GenericComponent.h"
 
#include "SealBase/Signal.h"
 
#include "SealKernel/Context.h"
#include "SealKernel/Exception.h"
 
#include <boost/bind.hpp>
#include <boost/signal.hpp>
#include <boost/signals/connection.hpp>
#include <boost/signals/trackable.hpp>
 
 
namespace frappe {
 
  template<typename Event>
  class Dispatcher : public boost::signal<void(Event const *)> {
  public:
    static void addDispatcher(frappe::Configurator &config);
  };
   
   
   
  template<typename Event>
  class  Fanout : public frappe::Client, public boost::signal<void(Event const *)> 
/*,  public boost::signals::trackable */ {
  public:
    typedef  boost::signal<void(Event const *)> super;
    Fanout(seal::Context* ic) :  frappe::Client(*ic) {
      // auto-register to Dispatcher
      m_conn = this->template component<Dispatcher<Event> >().connect(boost::bind(&super::operator(),this,_1));
    }
    ~Fanout() { m_conn.disconnect();}
     
     
    void reconnect() {
      m_conn.disconnect();
      m_conn = this->template component<Dispatcher<Event> >().connect(boost::bind(&super::operator(),this,_1));
    }
     
    boost::signals::connection m_conn;
  };
   
  template<typename Event>
  inline void Dispatcher<Event>::addDispatcher(frappe::Configurator &config)  {
    config.addComponent(new frappe::Dispatcher<Event> ());
    std::vector< seal::IHandle< frappe::LazyComponent<frappe::Fanout<Event> > > > matches;
    frappe::queryInChildren (config.context(), matches);
    for ( typename std::vector< seal::IHandle< frappe::LazyComponent<frappe::Fanout<Event> > > >::iterator
            p=matches.begin();p!=matches.end();p++)
      (**p)().reconnect();
  }

  template<typename Event>
  class Reflector : public frappe::Client, public boost::signals::trackable {
  public:
    typedef  Reflector<Event> self;
    Reflector(seal::Context* ic) :
      frappe::Client(*ic) {
      seal::Context * parent = context().parent();
      if (!parent) return;
      // auto-register into parent fanout
      frappe::Configurator conf(*parent);
      if (!conf.template exists<Fanout<Event> >()) conf.addComponent(new Fanout<Event>(parent));
      conf.template component<Fanout<Event> >().connect(boost::bind(&self::propagate,this,_1));
    }

    void propagate(Event const * ev) const {
      if ( exists<Dispatcher<Event> >() ) this->template component<Dispatcher<Event> >()(ev);
    }
  };

  template<typename Event>
  class Observer: public frappe::Client, public boost::signals::trackable {
  public:

    typedef  Observer<Event> self;
    Observer(seal::Context* ic) :
      frappe::Client(*ic) {}

    // to be called "after" child constructor
    void initialize() {
      // auto-register (first dump a local fanout)
      frappe::Configurator conf(context());
      if (!conf.template exists<Fanout<Event> >()) conf.addComponent(new Fanout<Event> (&context()));
      this->template component<Fanout<Event> >().connect(boost::bind(&self::callBack,this,_1));
    }

    void callBack(Event const * ev) const {
      const_cast<self*>(this)->update(ev);
    }

  private:
    virtual void update(Event const * ev) =0;

  };

}

struct MyMessage  : public std::string {
  explicit MyMessage(const std::string & s) : std::string (s) {}
};


class Echo : private frappe::Observer<MyMessage> {
public:
  typedef frappe::Observer<MyMessage> super;

  Echo(seal::Context* ic, const std::string& iname) :super(ic) {
    initialize();
  }

private:

  virtual void update(const MyMessage  * ev) {
    if (ev!=0) dump(*ev);
  }

  void dump(const  MyMessage & mess) const {
    std::cout << m_name << ": " << mess << std::endl;
  }

  std::string m_name;
};

// it owns the context...
class MyConfigurator {
public:
  MyConfigurator(seal::Context& iparent) :
    m_context(&iparent),  me(m_context){
  }

  MyConfigurator() : me(m_context){
  }

  seal::Context & context()  { return m_context;}

  void nameIt(const std::string& iname) {
    me.addComponent(new frappe::ContextName(iname));
  }

  // configurator depends on concrete stream, clients do not
  // void makeLocalStream() {
  //  me.addComponent<frappe::LogStreams>(ExMain::newCout( context(),me.component<frappe::ContextName>(),false));
  // }


  template<typename Event>
  void addDispatcher() {
    frappe::Dispatcher<Event>::addDispatcher(me);
  }

  template<typename Event>
  void addReflector() {
    me.addComponent(new frappe::Reflector<Event> (&context()));
  }

  // use name (parsing config file
  bool configure(const std::string & bname, const std::string & sname){
    return me.configure(bname,sname);
  }

private:
  seal::Context m_context;
  frappe::Configurator me;
};

#endif
