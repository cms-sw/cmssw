---
title: CMS Offline Software
layout: default
related:
 - { name: Project page, link: 'https://github.com/cms-sw/cmssw' }
 - { name: Feedback, link: 'https://github.com/cms-sw/cmssw/issues/new' }
---

## Tutorial: collaborating with peers.

This tutorial will guide you through the process of collaborating with other
people.

### Before you start.

Please make sure you registered to GitHub and that you have provided them
a ssh public key to access your private repository. For more information see
the [FAQ](faq.html).

### Development workflow

The development model CMS uses is to consider one repository:

https://github.com/cms-sw/cmssw

as the authoritative source of CMSSW sources. All other personal repositories
(aka forks, i.e. copies) are either for:

- Proposing changes to the official repository.
- Share changes between peers.

Proposing changes happens by creating a Pull Request from a given repository to
the official one (also referred as official-cmssw), this is documented in the
[Proposing changes to CMSSW tutorial](tutorial.html).

Sharing changes between peers is done instead by merging a remote branch in
your local workarea. Such a branch is not necessarily associated to a pull
request nor necessarily coming from the official CMSSW repository, effectively
allowing groups of people to collaborate with each other before a topic is
proposed for official inclusion in CMSSW.

### git cms-merge-topic

While checking out a topic can be done with the standard git commands: `git
fetch` + `git merge`, `git pull`, we provide an helper script, called

    git cms-merge-topic

which removes some of the boiler-plate required. This command can be used for
two different use cases:

- Merging a pull request:

      git cms-merge-topic <pull-request-id>

- Merging a remote branch:

      git cms-merge-topic <github-user>:<branch-name>

the latter command will use the official-cmssw remote in case no github
username is provided.

