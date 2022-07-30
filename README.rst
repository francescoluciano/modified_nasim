Network Attack Simulator - Modified by Francesco Lucianò
========================================================

|docs|

Network Attack Simulator (NASim) is a simulated computer network complete with vulnerabilities, scans and exploits designed to be used as a testing environment for AI agents and planning techniques applied to network penetration testing. This modified version implement the `PPO algorithm from stable_baselines3 <https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html>`_


Installation
------------

1. Clone the repository:
  ``$ git clone https://github.com/francescoluciano/modified_nasim.git``

2. Install the requirements:
  ``$ pip3 install -r requirements.txt``

3. Install python3-tk:
  ``$ sudo apt install python3-tk``


Quick start
-----------

To try the ppo_agent with the tiny scenario just move to the ``/nasim/agents`` directory and type

	$ ``python ppo_agent.py``

Look at the code of ``ppo_agent.py`` to change the parameters.

Documentation
-------------

The documentation of NAS is available at: https://networkattacksimulator.readthedocs.io/




NAS Authors
-------

**Jonathon Schwartz** - Jonathon.schwartz@anu.edu.au


License
-------

`MIT`_ © 2020, Jonathon Schwartz

.. _MIT: LICENSE


What's new
----------

- 2022-07-30 (First commit)

  + First commit

.. |docs| image:: https://readthedocs.org/projects/networkattacksimulator/badge/?version=latest
    :target: https://networkattacksimulator.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status
    :scale: 100%
