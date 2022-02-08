# CWRU Duckietown Driving School

This driving school system is developed by [Frank](mailto:frank_chude.qian@outlook.com) to better obtain human drivable logs for the duckietown system.

The system utilizes an Xbox 360 joystick to drive around. Left up and down controls the speed and right stick left and right controls the velocity. Right trigger enables the ["DRS" mode](https://en.wikipedia.org/wiki/Drag_reduction_system) allows vehicle to drive full speed forward. (Note there are no angular acceleration).

In addition, every 1500 steps in simulator, the recording will pause and playback. You will have the chance to review the result and decide whether to keep the log or not. The log are recorded into two formats: `raw_log` saves all the raw information for future re-processing, and `traning_data` saves the directly feedable log.

To run the script:

    $ python3 human.py

## Setup

To use this script, you will need duckietown gym installed.

In addition, you will need to instsall Ubuntu Xbox joystick drivers:

    $ sudo apt-get install xboxdrv
    $ sudo rmmod xpad
    $ sudo xboxdrv

## Folder file sctructure
