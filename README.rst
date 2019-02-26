=====
About
=====

This is a very simple face detection and recognition implementation in python.
It uses opencv to detect faces and
`eigenfaces <https://github.com/antonyross/eigenfaces>`_ to recognize the faces.

.. image:: demo.gif


Dependencies
============

Install dependecies::

   $ pip3 install -r requirements.txt

Usage
=====

First of all you have to train face recognizer how your face looks like.
Then you can test it with demo command.

Training
========

::

    $ python3 facerecognition/main.py train

This command will take 10 pictures using your webcam.
Press any key after every shot to take a new one.

Demo
====

To test face recognition use the demo that captures view from your webcam
and recognizes faces in realtime::

    $ python3 facerecognition/main.py demo

Requirements
============

* NumPy
* OpenCV 3
* matplotlib
* PIL - Python Imaging Library
* Python 3

Acknowledgements
================

Big thanks to Antony Ross providing me with the face recognition library
and guiding me through it.


.. rubric:: References

.. [#f1] http://www.vision.jhu.edu/teaching/vision08/Handouts/case_study_pca1.pdf
.. [#f2] http://wearables.cc.gatech.edu/paper_of_week/viola01rapid.pdf
