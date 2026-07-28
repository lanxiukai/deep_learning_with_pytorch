"""One-shot Matplotlib backend selection (private module).

Must be imported before ``matplotlib.pyplot`` (directly or via Matplotlib)
to take effect.  Import order between ``figures`` and ``images`` no longer
matters because both import this module first.

This module does **not** import ``figures``, ``images``, or ``pyplot``.

Policy
------
``MPLBACKEND`` environment variable is set
    -> do nothing (respect the user's explicit override).

No ``DISPLAY`` and no ``WAYLAND_DISPLAY``
    -> ``Agg`` (headless / pure-server).

``WSL_INTEROP`` is set and a display is available
    -> try ``TkAgg`` (works under WSLg); fall back to ``Agg`` if ``tkinter``
       cannot be imported.

Non-WSL graphical environment
    -> leave the backend to Matplotlib's own default (no intervention).
"""

import os

_MPLBACKEND = os.environ.get("MPLBACKEND")
if _MPLBACKEND is None:
    _display = os.environ.get("DISPLAY")
    _wayland = os.environ.get("WAYLAND_DISPLAY")
    if not _display and not _wayland:
        import matplotlib

        matplotlib.use("Agg")
    elif os.environ.get("WSL_INTEROP"):
        import matplotlib

        try:
            __import__("tkinter")
        except ImportError:
            matplotlib.use("Agg")
        else:
            matplotlib.use("TkAgg")
