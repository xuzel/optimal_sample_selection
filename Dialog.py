from kivy.factory import Factory
from kivy.properties import ObjectProperty
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.popup import Popup

from algorithms import *
from database import database
from database.database import Database


class DownloadDialog(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)
    cwdir = ObjectProperty(None)


class SelectDialog(FloatLayout):
    refilter = ObjectProperty(None)
    cancel = ObjectProperty(None)

class DataDialog(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)


Factory.register("DataDialog", cls=DataDialog)
Factory.register("SelectDialog", cls=SelectDialog)
Factory.register("DownloadDialog", cls=DownloadDialog)
