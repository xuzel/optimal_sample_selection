from kivy.app import App
from kivy.uix.floatlayout import FloatLayout


class IndexPage(FloatLayout):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    @staticmethod
    def page_database(*args):
        App.get_running_app().screen_manager.current = "Database_page"
        App.get_running_app().screen_manager.transition.direction = 'left'

    @staticmethod
    def page_select(*args):
        App.get_running_app().screen_manager.current = "Select_page"
        App.get_running_app().screen_manager.transition.direction = 'left'

    @staticmethod
    def page_refilter(*args):
        App.get_running_app().screen_manager.current = "Refilter_page"
        App.get_running_app().screen_manager.transition.direction = 'left'
