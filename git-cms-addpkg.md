---
title: CMS Offline Software
layout: default
related:
 - { name: Project page, link: 'https://github.com/cms-sw/cmssw' }
 - { name: Feedback, link: 'https://github.com/cms-sw/cmssw/issues/new' }
---
# git-cms-addpkg 

One very common development workflow in CMS is to checkout a few packages from
a base release in order to modify them. This is general done with the `addpkg`
wrapper around CVS.

A typical development day with CVS looks like:

    scram project CMSSW_6_1_0
    addpkg FWCore/Framework
    addpkg FWCore/Utilities
    <development>
    checkdeps -a

where `checkdeps` is used to bring in more packages that depend on the new
developments.

Git sees repository status globally, so the concept of checking out a single
package, or a single file, does not really exists. 

However a workaround to this is to use the so called ["sparse
checkout"](https://www.kernel.org/pub/software/scm/git/docs/git-read-tree.html#_sparse_checkout)
feature of git. This is controlled by the `.git/info/sparse-checkout` which
allows the user to specify which packages will have to be visible to the user
and which ones will not.

As a caveat, one must keep in mind that git still sees changes globally, so
packages being checked out will have to belong to the same tag. While this
might sound scary, it is not such a big issues given that git allows us to have
one tag per integration build. In the end, one could argue that per packages
tags were a workaround to the limitation imposed by cost of tags in CVS.
 
simply becomes:

    git cms-addpkg FWCore/Framework
    git cms-addpkg FWCore/Utilities
    <develop>
    git cms-checkdeps -a

For the above example the `git cms-addpkg` helper will simply create for you a
`.git/info/sparse-checkout` folder which contains:

    FWCore/Framework
    FWCore/Utilities     

merges the release tag into the current work area (notice the current
repository only contains a limited set of tags) and then invoke the command to
update the checkout area to match the contents of `.git/info/sparse-checkout`:

    git read-tree -mu HEAD

A final caveat is that it will not be possible to remove a package by hand via
`rm -fr <package-name>`, `git` will notice and will consider that as an actual
change. To hide a package, you will need to remove it from
`.git/info/sparse-checkout` and run `git read-tree -mu HEAD` again.

Note: do we need to provide a `git rmpkg` helper?
