# Examples

Inside this folder are a set of examples intended to show the basic interface when working with the gym-retro. Below are descriptions of the different example programs.

## basicGymRetroExample.py
Opens up street fighter 2 championship edition and plays against chunli. This example does random inputs every frame.

## frameByFrameTesting.py
Allows you to feed in specific inputs frame by frame of emulation to see the player character's reactions. This allows for testing things such as combo input buffering and ending lag. Does not use any custom input space discretizer.

## getActionMeaning.py
Runs through each specific index in the input space and returns a human readable description of what button is being pressed. Does not use any custom input space discretizer.

## customDiscretizerExample.py
Demonstrates how to build a custom discretization of the input space that works in human readable input specifications vs arrays of flags mapping to specific button presses.
