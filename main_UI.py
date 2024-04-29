from kivy.app import App
from kivy.core.window import Window
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.popup import Popup
from kivy.uix.screenmanager import Screen, ScreenManager
import sys

from Config import Config
from DatabasePage import DatabasePage
from IndexPage import IndexPage
from SelectPage import SelectPage
from RefilterPage import RefilterPage

import win32timezone

class Oss(App):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config =config

    def build(self):
        self.icon = "./static/logo.ico"
        self.title = "OSS App"

        self.load_kv("guipge/index.kv")  # create a index.kv file
        self.load_kv("guipge/select.kv")  # create a image.kv file
        self.load_kv("guipge/database.kv")  # create a image.kv file
        self.load_kv("guipge/refilter.kv")

        self.screen_manager = ScreenManager()
        Window.size = (1000, 720)
        # pages = {"Index_page": IndexPage(), "Database_page": DatabasePage()}
        pages = {"Index_page": IndexPage(), 
                "Select_page": SelectPage(self.config, 'GA'), 
                "Database_page": DatabasePage(self.config), 
                "Refilter_page":RefilterPage(self.config)}

        for item, page in pages.items():
            self.default_page = page
            # add page
            screen = Screen(name=item)
            screen.add_widget(self.default_page)
            # add page from screen manager
            self.screen_manager.add_widget(screen)
        return self.screen_manager


if __name__ == "__main__":
    sys.setrecursionlimit(3000)
    config = Config()
    Oss(config).run()
