"""Folder layout:  <root>/{fpl.py, prefs.json, app/, data/, output/}"""
import os
APP = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(APP)
DATA = os.path.join(ROOT, "data")
PLAYERS = os.path.join(DATA, "players")
OUT = os.path.join(ROOT, "output")
for _d in (DATA, PLAYERS, OUT):
    os.makedirs(_d, exist_ok=True)
