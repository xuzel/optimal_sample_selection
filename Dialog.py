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
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)
    input = ObjectProperty(None)
    config = ObjectProperty(None)

    def run_algorithms(self):
        # try:
        #     database = Database()
        #     self.config.result = list(database.find_one(str(self.config.m)+"-"+str(self.config.n)+"-"+str(self.config.k)+"-"+str(self.config.j)+"-"+str(self.config.s)))[0]

        # except:
        self.ids.result_output.text = "Waiting!!!"
        self.config.result = search(self.config.n, self.config.k, self.config.j, self.config.s)
        return self.config.result

    def import_database(self):
        input_format = str(self.config.m) + "-" + str(self.config.n) + "-" + \
                       str(self.config.k) + "-" + str(self.config.j) + "-" + str(self.config.s)
        database = Database()
        database.insert_one(input_format, self.config.result)
        return "Upload Successfully!!!!"

class DataDialog(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)


Factory.register("DataDialog", cls=DataDialog)
Factory.register("SelectDialog", cls=SelectDialog)
Factory.register("DownloadDialog", cls=DownloadDialog)
