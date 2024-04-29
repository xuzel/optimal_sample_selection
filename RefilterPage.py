from kivy.app import App
from kivy.uix.floatlayout import FloatLayout


class RefilterPage(FloatLayout):

    def __init__(self,config, **kwargs):
        super().__init__(**kwargs)
        self.config = config

    @staticmethod
    def page_index(*args):
        App.get_running_app().screen_manager.current = "Index_page"
        App.get_running_app().screen_manager.transition.direction = 'right'

    @staticmethod
    def page_database(*args):
        App.get_running_app().screen_manager.current = "Database_page"
        App.get_running_app().screen_manager.transition.direction = 'right'

    def update_box(self, text):
        self.ids.box1.opacity = 0
        self.ids.box2.opacity = 0
        self.ids.box3.opacity = 0
    
        if text == 'Define positions and numbers':
            self.ids.box1.opacity = 1
        elif text == 'Define positions and range of numbers':
            self.ids.box2.opacity = 1
        elif text == 'Select positions with fixed numbers':
            self.ids.box3.opacity = 1