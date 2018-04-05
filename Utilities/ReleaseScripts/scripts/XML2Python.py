## Original version of code heavily based on recipe written by Wai Yip
## Tung, released under PSF license.
## http://code.activestate.com/recipes/534109/

import re
import os
import xml.sax.handler

class DataNode (object):

    def __init__ (self, **kwargs):
        self._attrs = {}     # XML attributes and child elements
        self._data  = None   # child text data
        self._ncDict = kwargs.get ('nameChangeDict', {})

    def __len__ (self):
        # treat single element as a list of 1
        return 1

    def __getitem__ (self, key):
        if isinstance (key, basestring):
            return self._attrs.get(key,None)
        else:
            return [self][key]

    def __contains__ (self, name):
        return name in self._attrs

    def __nonzero__ (self):
        return bool (self._attrs or self._data)

    def __getattr__ (self, name):
        if name.startswith('__'):
            # need to do this for Python special methods???
            raise AttributeError (name)
        return self._attrs.get (name, None)

    def _add_xml_attr (self, name, value):
        change = self._ncDict.get (name)
        if change:
            name = change
        if name in self._attrs:
            # multiple attribute of the same name are represented by a list
            children = self._attrs[name]
            if not isinstance(children, list):
                children = [children]
                self._attrs[name] = children
            children.append(value)
        else:
            self._attrs[name] = value

    def __str__ (self):
        return self._data or ''

    def __repr__ (self):
        items = sorted (self._attrs.items())
        if self._data:
            items.append(('data', self._data))
        return u'{%s}' % ', '.join([u'%s:%s' % (k,repr(v)) for k,v in items])

    def attributes (self):
        return self._attrs


class TreeBuilder (xml.sax.handler.ContentHandler):

    non_id_char = re.compile('[^_0-9a-zA-Z]')

    def __init__ (self, **kwargs):
        self._stack = []
        self._text_parts = []
        self._ncDict = kwargs.get ('nameChangeDict', {})
        self._root = DataNode (nameChangeDict = self._ncDict)
        self.current = self._root

    def startElement (self, name, attrs):
        self._stack.append( (self.current, self._text_parts))
        self.current = DataNode (nameChangeDict = self._ncDict)
        self._text_parts = []
        # xml attributes --> python attributes
        for k, v in attrs.items():
            self.current._add_xml_attr (TreeBuilder._name_mangle(k), v)

    def endElement (self, name):
        text = ''.join (self._text_parts).strip()
        if text:
            self.current._data = text
        if self.current.attributes():
            obj = self.current
        else:
            # a text only node is simply represented by the string
            obj = text or ''
        self.current, self._text_parts = self._stack.pop()
        self.current._add_xml_attr (TreeBuilder._name_mangle(name), obj)

    def characters (self, content):
        self._text_parts.append(content)

    def root (self):
        return self._root

    def topLevel (self):
        '''Returns top level object'''
        return self._root.attributes().values()[0]
        

    @staticmethod
    def _name_mangle (name):
        return TreeBuilder.non_id_char.sub('_', name)


regexList = [ (re.compile (r'&'), '&amp;'   ),
              (re.compile (r'<'), '&lt;'    ),
              (re.compile (r'>'), '&gt;'    ),
              (re.compile (r'"'), '&quote;' ),
              (re.compile (r"'"), '&#39;'   )
              ]

quoteRE = re.compile (r'(\w\s*=\s*")([^"]+)"')

def fixQuoteValue (match):
    '''Changes all characters inside of the match'''
    quote = match.group(2)
    for regexTup in regexList:
        quote = regexTup[0].sub( regexTup[1], quote )
    return match.group(1) + quote + '"'


def xml2obj (**kwargs):
    ''' Converts XML data into native Python object.  Takes either
    file handle or string as input.  Does NOT fix illegal characters.

    input source:  Exactly one of the three following is needed
    filehandle     - input from file handle
    contents       - input from string
    filename       - input from filename

    options:
    filtering      - boolean value telling code whether or not to fileter
                     input selection to remove illegal XML characters
    nameChangeDict - dictionaries of names to change in python object'''

    # make sure we have exactly 1 input source
    filehandle = kwargs.get ('filehandle')
    contents   = kwargs.get ('contents')
    filename   = kwargs.get ('filename')
    if not filehandle and not contents and not filename:
        raise RuntimeError("You must provide 'filehandle', 'contents', or 'filename'")
    if     filehandle and contents or \
           filehandle and filename or \
           contents   and filename:
        raise RuntimeError("You must provide only ONE of 'filehandle', 'contents', or 'filename'")

    # are we filtering?
    filtering = kwargs.get ('filtering')
    if filtering:
        # if we are filtering, we need to read in the contents to modify them
        if not contents:
            if not filehandle:
                try:
                    filehandle = open (filename, 'r')
                except:
                    raise RuntimeError("Failed to open '%s'" % filename)
            contents = ''
            for line in filehandle:
                contents += line
            filehandle.close()
            filehandle = filename = ''
        contents = quoteRE.sub (fixQuoteValue, contents)
    
    ncDict = kwargs.get ('nameChangeDict', {})
    builder = TreeBuilder (nameChangeDict = ncDict)
    if contents:
        xml.sax.parseString(contents, builder)
    else:
        if not filehandle:
            try:
                filehandle = open (filename, 'r')
            except:
                raise RuntimeError("Failed to open '%s'" % filename)
        xml.sax.parse(filehandle, builder)
    return builder.topLevel()
