/*********************************************************
* $Id: RPXMLConfig.cc,v 1.1.1.1 2007/05/16 15:44:49 hniewiad Exp $
* $Revision: 1.1.1.1 $
* $Date: 2007/05/16 15:44:49 $
**********************************************************/

#include "RPXMLConfig.h"

// -------------------------------------------------------
// Class RPXMLConfig
// -------------------------------------------------------

using namespace std;



RPXMLConfig::RPXMLConfig ()
{

  try
  {
    XMLPlatformUtils::Initialize ();
  }
  catch (const XMLException & toCatch)
  {
    char *message = XMLString::transcode (toCatch.getMessage ());
    std::cout << "Error during initialization! :\n" << message << "\n";
    XMLString::release (&message);
  }

  XMLCh tempStr[100];
  XMLString::transcode ("LS", tempStr, 99);
  DOMImplementation *impl = DOMImplementationRegistry::getDOMImplementation (tempStr);
  this->parser =  new XercesDOMParser();
  this->writer = ((DOMImplementationLS*)impl)->createLSSerializer();
//  ((DOMImplementationLS *) impl)->createDOMWriter ();

  if (this->writer->getDomConfig()->canSetParameter (XMLUni::fgDOMWRTFormatPrettyPrint, true))
    this->writer->getDomConfig()->setParameter (XMLUni::fgDOMWRTFormatPrettyPrint, true);

  this->doc = 0;

  this->id_string = XMLString::transcode ("id");
  this->subitem_string = XMLString::transcode ("subitem");
  this->item_string = XMLString::transcode ("item");
  this->config_string = XMLString::transcode ("config");

}

RPXMLConfig::~RPXMLConfig ()
{
//  this->parser->release ();
  XMLString::release (&this->id_string);
  XMLString::release (&this->subitem_string);
  XMLString::release (&this->item_string);
  XMLString::release (&this->config_string);
}



// -------------------------------------------------------
// Read & write
// -------------------------------------------------------

void
RPXMLConfig::save ()
{
  this->save (this->fileName);
}


void
RPXMLConfig::save (const std::string filename)
{

  // Convert the path into Xerces compatible XMLCh*.
    XMLCh *tempFilePath = XMLString::transcode(filename.c_str());

    // Calculate the length of the string.
    const int pathLen = XMLString::stringLen(tempFilePath);

    // Allocate memory for a Xerces string sufficent to hold the path.
    XMLCh *targetPath = (XMLCh*)XMLPlatformUtils::fgMemoryManager->allocate((pathLen + 9) * sizeof(XMLCh));

    // Fixes a platform dependent absolute path filename to standard URI form.
    XMLString::fixURI(tempFilePath, targetPath);

    // Specify the target for the XML output.
    XMLFormatTarget *formatTarget = new LocalFileFormatTarget(targetPath);
    //XMLFormatTarget *myFormTarget = new StdOutFormatTarget();

    // Create a new empty output destination object.
    XMLCh tempStr[100];
    XMLString::transcode ("LS", tempStr, 99);
    DOMImplementation *impl = DOMImplementationRegistry::getDOMImplementation (tempStr);
    DOMLSOutput *output = ((DOMImplementationLS*)impl)->createLSOutput();

    // Set the stream to our target.
    output->setByteStream(formatTarget);

   //XMLFormatTarget *myForm = new LocalFileFormatTarget(filename.c_str());
   //XMLFormatTarget *myForm = new StdOutFormatTarget ();
    writer->write (doc, output);

      // Cleanup.
    writer->release();
    XMLString::release(&tempFilePath);
    delete formatTarget;
    output->release();
}

void
RPXMLConfig::read ()
{
  this->read (this->fileName);
}

void
RPXMLConfig::parse (const std::string filename)
{
  try
  {
//    char * cstr = new char [filename.length()+1];
    this->parser->parse(filename.c_str());
    this->doc = this->parser->getDocument ();
  }
  catch (const XMLException & toCatch)
  {
    char *message = XMLString::transcode (toCatch.getMessage ());
    std::cerr << "Exception message is: \n" << message << "\n";
    XMLString::release (&message);
  }
  catch (const DOMException & toCatch)
  {
    char *message = XMLString::transcode (toCatch.msg);
    std::cerr << "Exception message is: \n" << message << "\n";
    XMLString::release (&message);
  }
  catch (...)
  {
    std::cerr << "Unexpected Exception \n";
  }
}


void
RPXMLConfig::read (const std::string filename)
{

  this->parse (filename);

  if (this->isEmpty ())
    {
      DOMElement *e = this->doc->createElement (this->config_string);
      this->doc->appendChild (e);
    }

  this->mn = this->doc->getFirstChild ();

}


// -------------------------------------------------------
// Getters & setters
// -------------------------------------------------------

void
RPXMLConfig::setFilename (std::string filename)
{
  this->fileName = filename;
}

std::string RPXMLConfig::getFilename ()
{
  return this->fileName;
}


// -------------------------------------------------------
// First level access
// -------------------------------------------------------

vector<int>
RPXMLConfig::getIds()
{
  vector<int> v;
  unsigned int i;
  int id;
  DOMNodeList *nl = mn->getChildNodes ();
  for (i = 0; i < nl->getLength (); i++)
    {
      if (XMLString::equals (nl->item (i)->getNodeName (), this->item_string))
	{
	  id = XMLString::parseInt (nl->item (i)->getAttributes ()->
				    getNamedItem (this->id_string)->
				    getNodeValue ());
	  v.push_back ( id );
	}
    }
  return v;
}


void
RPXMLConfig::set (int id, const char *name, const char *value)
{
  unsigned int i;

  XMLCh *attr_name = XMLString::transcode (name);
  XMLCh *attr_value = XMLString::transcode (value);

  // adding <config> node, if missing
  if (this->isEmpty ())
    {
      DOMElement *e = this->doc->createElement (this->config_string);
      this->doc->appendChild (e);
    }
  // adding <item> node, if missing
  if (!this->containItem (id))
    {
      this->addItem (id);
    }

  DOMNodeList *nl = mn->getChildNodes ();

  // loop over all <item> nodes
  for (i = 0; i < nl->getLength (); i++)
    {
      // check if node is <item> node
      if (XMLString::equals (nl->item (i)->getNodeName (), this->item_string))
	{
	  // check if it has appriorate id value
	  if (XMLString::
	      parseInt (nl->item (i)->getAttributes ()->
			getNamedItem (this->id_string)->getNodeValue ()) ==
	      id)
	    ((DOMElement *) (nl->item (i)))->setAttribute (attr_name,
							   attr_value);
	}
    }
}

char *
RPXMLConfig::get (int id, const char *name)
{
  unsigned int i;
  DOMNodeList *nl = mn->getChildNodes ();
  XMLCh *attr_name = XMLString::transcode (name);
  for (i = 0; i < nl->getLength (); i++)
    {
      if (XMLString::equals (nl->item (i)->getNodeName (), this->item_string))
	{
	  if (XMLString::
	      parseInt (nl->item (i)->getAttributes ()->
			getNamedItem (this->id_string)->getNodeValue ()) ==
	      id)
	    if (nl->item (i)->getAttributes ()->
		getNamedItem (attr_name) != NULL)
	      return XMLString::transcode (nl->item (i)->
					   getAttributes ()->
					   getNamedItem (attr_name)->
					   getNodeValue ());
	    else
	      return NULL;
	}
    }
  return NULL;
}

int
RPXMLConfig::getInt (int id, char *name)
{
  if (this->get (id, name) != NULL)
    return atoi (this->get (id, name));
  else
    return 0;
}

void
RPXMLConfig::setInt (int id, char *name, int value)
{
  std::stringstream s;
  s << value;
  this->set (id, name, s.str ().c_str ());
}

double
RPXMLConfig::getDouble (int id, char *name)
{
  if (this->get (id, name) != NULL)
    return atof (this->get (id, name));
  else
    return 0;
}

void
RPXMLConfig::setDouble (int id, char *name, double value)
{
  std::stringstream s;
  s << value;
  this->set (id, name, s.str ().c_str ());
}


// -------------------------------------------------------
// Second level access 
// -------------------------------------------------------

vector<int>
RPXMLConfig::getIds(int id1)
{
  vector<int> v;
  int id2;
  unsigned int i, j;
  DOMNodeList *l1 = this->mn->getChildNodes ();
  DOMNodeList *l2;
  for (i = 0; i < l1->getLength (); i++)
    {
      if (XMLString::equals (l1->item (i)->getNodeName (), this->item_string))
	{
	  if (XMLString::
	      parseInt (l1->item (i)->getAttributes ()->
			getNamedItem (this->id_string)->getNodeValue ()) ==
	      id1)
	    {
	      l2 = l1->item (i)->getChildNodes ();
	      for (j = 0; j < l2->getLength (); j++)
		{
		  if (XMLString::
		      equals (l2->item (j)->getNodeName (),
			      this->subitem_string))
		    {
		      id2 = XMLString::
			parseInt (l2->item (j)->getAttributes ()->
				  getNamedItem (this->id_string)->
				  getNodeValue ());
		      v.push_back (id2);
		    }
		}
	    }
	}
    }
  return v;
}

void
RPXMLConfig::set (int id1, int id2, const char *name, const char *value)
{
  unsigned int i,j;

  XMLCh *attr_name = XMLString::transcode (name);
  XMLCh *attr_value = XMLString::transcode (value);

  // adding <config> node, if missing
  if (this->isEmpty ())
    {
      DOMElement *e = this->doc->createElement (this->config_string);
      this->doc->appendChild (e);
    }
  // adding <item> node, if missing
  if (!this->containItem (id1))
    {
      this->addItem (id1);
    }
  // adding <subitem> node, if missing
  if (!this->containSubItem (id1, id2))
    {
      this->addSubItem (id1, id2);
    }

  DOMNodeList *l1 = mn->getChildNodes ();
  DOMNodeList *l2;


  // loop over all <item> nodes
  for (i = 0; i < l1->getLength (); i++)
    {
      // check if node is <item> node
      if (XMLString::equals (l1->item (i)->getNodeName (), this->item_string))
	{
	  // check if it has appriorate id value
	  if (XMLString::
	      parseInt (l1->item (i)->getAttributes ()->
			getNamedItem (this->id_string)->getNodeValue ()) ==
	      id1)
	    {
	      // 2nd level access
	      l2 = l1->item (i)->getChildNodes ();
	      // loop over all <subitem> nodes
	      for (j = 0; j < l2->getLength (); j++)
		{
		  // check if node is <subitem> node
		  if (XMLString::
		      equals (l2->item (j)->getNodeName (),
			      this->subitem_string))
		    {
		      // check if it has appriorate id value
		      if (XMLString::
			  parseInt (l2->item (j)->getAttributes ()->
				    getNamedItem (this->id_string)->
				    getNodeValue ()) == id2)
			{
			  ((DOMElement *) (l2->item (j)))->
			    setAttribute (attr_name, attr_value);
			}
		    }
		}
	    }
	}
    }
}

char *
RPXMLConfig::get (int id1, int id2, const char *name)
{
  unsigned int i, j;
  DOMNodeList *l1 = this->mn->getChildNodes ();
  DOMNodeList *l2;
  XMLCh *attr_name = XMLString::transcode (name);
  for (i = 0; i < l1->getLength (); i++)
    {
      if (XMLString::equals (l1->item (i)->getNodeName (), this->item_string))
	{
	  if (XMLString::
	      parseInt (l1->item (i)->getAttributes ()->
			getNamedItem (this->id_string)->getNodeValue ()) ==
	      id1)
	    {
	      l2 = l1->item (i)->getChildNodes ();
	      for (j = 0; j < l2->getLength (); j++)
		{
		  if (XMLString::
		      equals (l2->item (j)->getNodeName (),
			      this->subitem_string))
		    {
		      if (XMLString::
			  parseInt (l2->item (j)->getAttributes ()->
				    getNamedItem (this->id_string)->
				    getNodeValue ()) == id2)
			{
			  if (l2->item (j)->getAttributes ()->
			      getNamedItem (attr_name) != NULL)
			    return XMLString::transcode (l2->item (j)->
							 getAttributes ()->
							 getNamedItem
							 (attr_name)->
							 getNodeValue ());
			  else
			    return NULL;
			}
		    }
		}
	    }
	}
    }
  return NULL;
}

int
RPXMLConfig::getInt (int id1, int id2, char *name)
{
  if (this->get (id1, id2, name) != NULL)
    return atoi (this->get (id1, id2, name));
  else
    return 0;
}

void
RPXMLConfig::setInt (int id1, int id2, char *name, int value)
{
  std::stringstream s;
  s << value;
  this->set (id1, id2, name, s.str ().c_str ());
}

double
RPXMLConfig::getDouble (int id1, int id2, char *name)
{
  if (this->get (id1, id2, name) != NULL)
    return atof (this->get (id1, id2, name));
  else
    return 0.0;
}

void
RPXMLConfig::setDouble (int id1, int id2, char *name, double value)
{
  std::stringstream s;
  s << value;
  this->set (id1, id2, name, s.str ().c_str ());
}

// -------------------------------------------------------
// Others 
// -------------------------------------------------------

bool
RPXMLConfig::isEmpty ()
{
  if (this->doc->hasChildNodes ())
    {
      return !XMLString::equals (this->doc->getFirstChild ()->
				 getNodeName (), this->config_string);
    }
  return false;
}


bool
RPXMLConfig::containItem (const int id)
{
  unsigned int i;
  DOMNodeList *nl;

  nl = this->mn->getChildNodes ();
  for (i = 0; i < nl->getLength (); i++)
    {
      if (XMLString::equals (nl->item (i)->getNodeName (), this->item_string))
	{
	  if (XMLString::
	      parseInt (nl->item (i)->getAttributes ()->
			getNamedItem (this->id_string)->getNodeValue ()) ==
	      id)
	    return true;
	}
    }
  return false;
}


void
RPXMLConfig::addItem (const int id)
{
  if (!this->containItem (id))
    {
      DOMElement *item = doc->createElement (this->item_string);
      std::stringstream s;
      s << id;
      item->setAttribute (this->id_string,
			  XMLString::transcode (s.str ().c_str ()));
      this->mn->appendChild (item);
    }
}


bool
RPXMLConfig::containSubItem (const int id1, const int id2)
{
  unsigned int i, j;
  DOMNodeList *l1;
  DOMNodeList *l2;

  l1 = this->mn->getChildNodes ();
  for (i = 0; i < l1->getLength (); i++)
    {
      if (XMLString::equals (l1->item (i)->getNodeName (), this->item_string))
	{
	  if (XMLString::
	      parseInt (l1->item (i)->getAttributes ()->
			getNamedItem (this->id_string)->getNodeValue ()) ==
	      id1)
	    {
	      l2 = l1->item (i)->getChildNodes ();
	      for (j = 0; j < l2->getLength (); j++)
		{
		  if (XMLString::
		      equals (l2->item (j)->getNodeName (),
			      this->item_string))
		    {
		      if (XMLString::
			  parseInt (l2->item (j)->getAttributes ()->
				    getNamedItem (this->id_string)->
				    getNodeValue ()) == id2)
			return true;
		    }
		}

	    }
	}
    }
  return false;
}

void
RPXMLConfig::addSubItem (const int id1, const int id2)
{
  if (!this->containSubItem (id1, id2))
    {
      DOMElement *item = doc->createElement (this->subitem_string);
      std::stringstream s;
      s << id2;
      item->setAttribute (this->id_string,
			  XMLString::transcode (s.str ().c_str ()));
      unsigned int i;
      DOMNodeList *nl;

      nl = this->mn->getChildNodes ();
      for (i = 0; i < nl->getLength (); i++)
	{
	  if (XMLString::
	      equals (nl->item (i)->getNodeName (), this->item_string))
	    {
	      if (XMLString::
		  parseInt (nl->item (i)->getAttributes ()->
			    getNamedItem (this->id_string)->
			    getNodeValue ()) == id1)
		nl->item (i)->appendChild (item);
	    }
	}
    }
}

bool
RPXMLConfig::containAttr (const int id, const char *name)
{
  unsigned int i;
  XMLCh *name_string = XMLString::transcode (name);

  DOMNodeList *nl = mn->getChildNodes ();
  for (i = 0; i < nl->getLength (); i++)
    {
      if (XMLString::equals (nl->item (i)->getNodeName (), this->item_string))
	{
	  if (XMLString::
	      parseInt (nl->item (i)->getAttributes ()->
			getNamedItem (this->id_string)->getNodeValue ()) ==
	      id)
	    {
	      if (nl->item (i)->getAttributes ()->
		  getNamedItem (name_string) != NULL)
		return true;
	    }
	}
    }
  return false;
}
