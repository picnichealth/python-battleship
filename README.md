# battleship_engine
A minimal battleship game engine.

Requirements:
  - python 3.6+
  - numpy
  - a terminal with unicode support

Run with:
```
  pip3 install -U numpy
  python3 -m battleship.game
```

You can also typecheck:
```
  pip3 install -U mypy
  mypy battleship
```
or run tests:
```
  pip3 install -U pytest
  pytest .
```

The game should look something like:

![Example Game](https://raw.githubusercontent.com/benpastel/battleship_engine/master/example_game.jpg?token=AAIN6GTAA76XQVAVLDGQCXC7J6MFE)

The interface assumes that `🚢💦💥` are exactly twice the width of ascii in your terminal output; otherwise it might look lopsided.

Happy hacking!
