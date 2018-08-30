"""CMS TagCollector Python API
"""

__author__ = "Miguel Ojeda"
__copyright__ = "Copyright 2010-2011, CERN CMS"
__credits__ = ["Miguel Ojeda"]
__license__ = "Unknown"
__maintainer__ = "Miguel Ojeda"
__email__ = "mojedasa@cern.ch"
__status__ = "Staging"

_tagcollector_url = 'https://cmstags.cern.ch/tc/'

import urllib
import urllib2
import cookielib
try:
    import json
except ImportError:
    import simplejson as json
import getpass
import ws_sso_content_reader

class TagCollector(object):
    """CMS TagCollector Python API"""

    def __init__(self,useCert=False):
        self._url = _tagcollector_url
        self.useCert = useCert
        self.login = False
        if self.useCert:
            self.login = True
            self._cj = cookielib.CookieJar()
            self._opener = urllib2.build_opener(urllib2.HTTPCookieProcessor(self._cj))

    def __del__(self):
        if self.login: self.signOut()

    def _open(self, page, params = None, data = None):
        url = self._url + page + '?'
        if params:
            url += urllib.urlencode(params)
        if data:
            data = urllib.urlencode(data)
        if self.useCert:
            return ws_sso_content_reader.getContent(url, '~/.globus/usercert.pem', '~/.globus/userkey.pem', data)
        try:
            return self._opener.open(url, data).read()
        except urllib2.HTTPError as e:
            raise Exception(e.read().strip())

    def _openjson(self, page, params = None, data = None):
        return json.loads(self._open(page, params, data))

    def signIn(self, username, password):
        if self.useCert: return
        """Sign in to TagCollector."""
        self._open('signIn', data = {'password': password, 'user_name': username})
        self.login = True

    def signInInteractive(self):
        if self.useCert: return
        """Sign in to TagCollector, asking for the username and password."""
        username = raw_input('Username: ')
        password = getpass.getpass()
        self.signIn(username, password)
        return username

    def signOut(self):
        if self.useCert: return
        """Sign out of TagCollector."""
        self._open('signOut')
        self.login = False

    def getPackageTags(self, package):
        """Get the tags published in TagCollector for a package.
        Note: TagCollector's published tags are a subset of CVS' tags."""
        return self._openjson('public/py_getPackageTags', {'package': package})

    def getPackageTagDescriptionFirstLine(self, package, tag):
        """Get the first line of the descriptions of a tag."""
        return self._openjson('public/py_getPackageTagDescriptionFirstLine', {'package': package, 'tag': tag})

    def getPackageTagReleases(self, package, tag):
        """Get the releases where a tag is."""
        return self._openjson('public/py_getPackageTagReleases', {'package': package, 'tag': tag})

    def getReleasesTags(self, releases, diff = False):
        """Get the tags of one or more release.
        Optionally, return only the tags that differ between releases."""
        releases = json.dumps(releases)
        diff = json.dumps(diff)
        return self._openjson('public/py_getReleasesTags', {'releases': releases, 'diff': diff})

    def getReleaseTags(self, release):
        """Get the tags of one release."""
        return self.getReleasesTags((release, ))

    def getTagsetTags(self, tagset):
        """Get the tags of one tagset."""
        return self._openjson('public/py_getTagsetTags', {'tagset': tagset})

    def getTagsetInformation(self, tagset):
        """Get the information of one tagset."""
        return self._openjson('public/py_getTagsetInformation', {'tagset': tagset})

    def getPendingApprovalTags(self, args, allow_multiple_tags = False):
        """Prints Pending Approval tags of one or more releases,
        one or more tagsets, or both (i.e. it joins all the tags).
        Prints an error if several tags appear for a single package.
        Suitable for piping to addpkg (note: at the moment,
        addpkg does not read from stdin, use "-f" instead)."""
        args = json.dumps(args)
        allow_multiple_tags = json.dumps(allow_multiple_tags)
        return self._openjson('public/py_getPendingApprovalTags', {'args': args, 'allow_multiple_tags': allow_multiple_tags})

    def getTagsetsTagsPendingSignatures(self, user_name, show_all, author_tagsets, release_names = None):
        """Prints Pending Signature tags of one or more releases,
        one or more tagsets, or both (i.e. it joins all the tags).
        Prints an error if several tags appear for a single package.
        Suitable for piping to addpkg (note: at the moment,
        addpkg does not read from stdin, use "-f" instead)."""
        if not release_names == None:
            return self._openjson('public/py_getTagsetsTagsPendingSignatures', {'user_name': user_name, 'show_all': show_all, 'author_tagsets': author_tagsets, 'release_names': json.dumps(release_names)})
        else:
            return self._openjson('public/py_getTagsetsTagsPendingSignatures', {'user_name': user_name, 'show_all': show_all, 'author_tagsets': author_tagsets})

    def commentTagsets(self, tagset_ids, comment):
        """Comment one or more tagsets.
        Requirement: Signed in."""
        tagset_ids = json.dumps(tagset_ids)
        if len(comment) < 1:
            raise Exception("Error: Expected a comment.")
        self._open('commentTagsets', {'tagset_ids': tagset_ids, 'comment': comment})

    def signTagsets(self, tagset_ids, comment = ''):
        """Sign one or more tagsets.
        Requirement: Signed in as a L2."""
        tagset_ids = json.dumps(tagset_ids)
        self._open('signTagsets', {'tagset_ids': tagset_ids, 'comment': comment})

    def signTagsetsAll(self, tagset_ids, comment = ''):
        """Sign all one or more tagsets.
        Requirement: Signed in as a top-level admin."""
        tagset_ids = json.dumps(tagset_ids)
        self._open('signTagsetsAll', {'tagset_ids': tagset_ids, 'comment': comment})

    def rejectTagsetsPendingSignatures(self, tagset_ids, comment = ''):
        """Reject one or more tagsets Pending Signatures.
        Requirement: Signed in as a L2s or as a Release Manager
        for the tagset's release or as the author of the tagset."""
        tagset_ids = json.dumps(tagset_ids)
        self._open('rejectTagsetsPendingSignatures', {'tagset_ids': tagset_ids, 'comment': comment})

    def approveTagsets(self, tagset_ids, comment = ''):
        """Approve one or more tagsets.
        Requirement: Signed in as a Release Manager for each tagset's release."""
        tagset_ids = json.dumps(tagset_ids)
        self._open('approveTagsets', {'tagset_ids': tagset_ids, 'comment': comment})

    def bypassTagsets(self, tagset_ids, comment = ''):
        """Bypass one or more tagsets.
        Requirement: Signed in as a Release Manager for each tagset's release."""
        tagset_ids = json.dumps(tagset_ids)
        self._open('bypassTagsets', {'tagset_ids': tagset_ids, 'comment': comment})

    def rejectTagsetsPendingApproval(self, tagset_ids, comment = ''):
        """Reject one or more tagsets Pending Approval.
        Requirement: Signed in as a Release Manager."""
        tagset_ids = json.dumps(tagset_ids)
        self._open('rejectTagsetsPendingApproval', {'tagset_ids': tagset_ids, 'comment': comment})

    def removeTagsets(self, tagset_ids, comment = ''):
        """Remove one or more tagsets from the History (i.e. stack of the release).
        Requirement: Signed in as a Release Manager."""
        tagset_ids = json.dumps(tagset_ids)
        self._open('removeTagsets', {'tagset_ids': tagset_ids, 'comment': comment})

    def getPackagesPendingApproval(self):
        """Get New Package Requests which are Pending Approval."""
        return self._openjson('public/py_getPackagesPendingApproval')

    def getPackageManagersRequested(self, package):
        """Get the Package Managers (administrators and developers) requested in a New Package Request."""
        return self._openjson('public/py_getPackageManagersRequested', {'package': package})

    def search(self, term):
        """Searches for releases, packages, tagsets, users and categories.
        Requirement: Signed in."""
        return self._openjson('search', {'term': term})

    def approvePackage(self, package):
        """Approve a New Package Request.
        Requirement: Signed in as a Creator (i.e. people in the top-level .admin/developers file).
        Warning: This does *not* create the package in CVS."""
        self._open('approveNewPackageRequest', {'package_name': package})

    def getIBs(self, filt = '', limit = 10):
        """Get the name and creation date of Integration Builds.
        By default, it only returns the latest 10 IBs.
        Optionally, filter by name."""
        return self._openjson('public/py_getIBs', {'filt': filt, 'limit': limit})

    def deprecateReleases(self, *releases):
        """ Deprecate releases"""
        if not self.login:
            raise Exception("Error: Not logged in?!")
        self._open('deprecateReleases', {"releases": ",".join(releases)})

    def createRelease(self, base_release_name, new_release_name, new_state, new_private, new_type, new_description, release_managers, copy_queues, tags):
        """Create a new release.
        Requirement: Signed in as a release manager."""
        if self.login:
            self._open('copyRelease', {'release_name': base_release_name, 'new_release_name': new_release_name, 'new_state': new_state, 'new_private': new_private, 'new_type': new_type, 'new_description': new_description, 'release_managers': release_managers, 'copy_queues': copy_queues, 'tags': tags})
        else:
            raise Exception("Error: Not logged in?!")

    def requestCustomIB(self, release_name, architectures, tags):
        """Request a CustomIB.
        Requirement: Signed in."""
        if self.login:
            self._open('requestCustomIB', {'release_name': release_name, 'architecture_names': architectures, 'tags': tags})
        else:
            raise Exception("Error: Not logged in?!")

    def getReleaseArchitectures(self, release, default='0'):
        """Returns release architectures."""
        return self._openjson('public/py_getReleaseArchitectures', {'release': release, 'default': default})
