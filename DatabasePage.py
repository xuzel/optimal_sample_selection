import logging

from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.popup import Popup

from Dialog import DownloadDialog
from our_atabase import Database


class DatabasePage(FloatLayout):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config

    @staticmethod
    def page_index(*args):
        App.get_running_app().screen_manager.current = "Index_page"
        App.get_running_app().screen_manager.transition.direction = 'right'

    @staticmethod
    def page_select(*args):
        App.get_running_app().screen_manager.current = "Select_page"
        App.get_running_app().screen_manager.transition.direction = 'right'

    @staticmethod
    def page_refilter(*args):
        App.get_running_app().screen_manager.current = "Refilter_page"
        App.get_running_app().screen_manager.transition.direction = 'left'

    # def search(self,m,n,k,j,s,alg):
    #     Database.search(m=m,n=n,k=k,j=j,s=s,alg=alg)

    def search(self, param):
        output = Database.search(**param)
        output_str = ""
        for key, value in output.items():
            print(key)
            output_str += (f"{key}\n"
                           f"run time is: {value['time']:.4f}\n"
                           f"the number of the solution is: {value['num_solution']}\n")
            for solution in value['solution']:
                output_str += f"{solution} \n"
        output_str += '\n\n'
        return output_str
